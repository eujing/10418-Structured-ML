import torch
import torch.nn as nn

from torch.utils.data import Dataset
from collections import defaultdict, namedtuple
import random, pdb
from nltk.tree import Tree
import numpy as np
import json
import datetime as dt
import argparse
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tag Dictionary
tag_dict = defaultdict(int)

def check_tag_exists(tag):
    if tag not in tag_dict:
        tag_val = max(tag_dict.values(), default=-1) + 1
        tag_dict[tag] = tag_val

def tag_to_vec(tag):
    n = len(tag_dict)
    tag_val = tag_dict[tag]
    dist = torch.zeros(n)
    dist[tag_val] = 1

    return dist

# Word Dictionary
word_dict = defaultdict(int)

def check_word_exists(word):
    if word not in word_dict:
        word_val = max(word_dict.values(), default=-1) + 1
        word_dict[word] = word_val


class OverallIndex():
    def __init__(self, val):
        self.val = val


class MyNode():
    def __init__(self, treeNode, parent):
        tag = treeNode._label

        check_tag_exists(tag)

        self.true_label = np.array([tag_dict[tag]])
        self.true_label = torch.tensor(self.true_label).long()

        # Assume not leaf first
        self.is_leaf = False
        self.children = []
        self.parent = parent

        for child in treeNode:
            if not isinstance(child, str):
                self.children.append(MyNode(child, self))

        # Is leaf if has no children
        if len(self.children) == 0:
            self.is_leaf = True
            word = treeNode[0]
            check_word_exists(word)
            self.word = word


def build_tree(tree):
    my_tree = MyNode(tree, None)
    return my_tree

def tree_get_leaves(node):
    if node.is_leaf:
        return [node]
    else:
        childrens_leaves = [tree_get_leaves(c) for c in node.children]
        # Flattens out list of lists in order
        return sum(childrens_leaves, [])

def tree_in_postorder(node):
    if node.is_leaf:
        return [node]
    else:
        children_results = [tree_in_postorder(c) for c in node.children]
        # Visit node after all its children
        acc = sum(children_results, []) + [node]
        return acc

def tree_in_preorder(node):
    if node.is_leaf:
        return [node]
    else:
        children_results = [tree_in_preorder(c) for c in node.children]
        # Visit node before all its children
        acc = sum(children_results, [node])
        return acc

def accuracy(preds, actuals):
    correct = (preds == actuals).sum().item()
    return correct / len(actuals)

# tree.leaves() gives the terminals for each tree.

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_size=256, hidden_size=128):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size)

    # Input is (L)
    def forward(self, input):
        L = len(input)

        # emb is L x 1 x emb_size
        emb = self.embedding(input).reshape((L, 1, self.embedding_size))

        # outputs is L x 1 x hidden_size
        outputs, _ = self.lstm(emb)

        # output is now L x hidden_size
        return outputs.squeeze()

class Baseline(nn.Module):
    def __init__(self, vocab_size, tag_size, hidden_size=128):
        super(Baseline, self).__init__()
        self.vocab_size = vocab_size
        self.tag_size = tag_size

        self.encoder = Encoder(vocab_size, hidden_size=hidden_size).to(device)

        # For pairwise combination of hidden outputs
        self.pair_to_hidden = nn.Linear(2 * hidden_size, hidden_size)

        # From hidden output to tag space
        self.hidden_to_tags = nn.Linear(hidden_size, tag_size)

    # Helper function to make tag predictions by traversing the tree
    # Params:
    #   node: Starting MyNode to make a predicition for
    #   leaf_to_hidden: Dictionary mapping a leaf MyNode to its hidden vector
    # Return:
    #   (curr_hidden, curr_tag, all_tags)
    #   curr_hidden: Hidden vector for given node
    #   curr_tag: Tag scores (pre-softmax) for given node
    #   all_tags: List of tag scores (pre-softmax) in post-order traversal
    #             of sub-tree starting from given node
    def forward_node(self, node, leaf_to_hidden):
        # Node is a leaf
        if node.is_leaf:
            curr_hidden = leaf_to_hidden[node]
            curr_tag = self.hidden_to_tags(curr_hidden)

            return curr_hidden, curr_tag, [curr_tag]

        # Interior node
        else:
            assert len(node.children) > 0
            assert len(node.children) <= 2

            child_left, child_right = node.children[0], node.children[-1]

            # Calculate children
            # Hidden are vectors of hidden_size
            # Tags are vectors of tag_size
            left_hidden, left_tag, left_acc = self.forward_node(child_left, leaf_to_hidden)
            right_hidden, right_tag, right_acc = self.forward_node(child_right, leaf_to_hidden)

            # Calculate current
            curr_hidden = self.pair_to_hidden(torch.cat([left_hidden, right_hidden], dim=0))
            curr_tag = self.hidden_to_tags(curr_hidden)

            # Accumulator of tag vectors is in post-order traversal
            # If only 1 child, left and right are the same so dont overcount
            if child_left == child_right:
                acc = left_acc + [curr_tag]
            else:
                acc = left_acc + right_acc + [curr_tag]

            return curr_hidden, curr_tag, acc

    # Assume that all trees are binary
    def forward(self, root):
        leave_nodes = tree_get_leaves(root)

        # Leaves input is vector of L
        leaves_input = torch.tensor(
            [word_dict[node.word] for node in leave_nodes],
            device=device)

        # Leaves hidden is L x hidden_size
        # But sometimes there is only 1 leaf

        leaves_hidden = self.encoder(leaves_input)
        if len(leave_nodes) == 1:
            assert len(leaves_hidden.shape) == 1
            leaf_to_hidden = {node: leaves_hidden for i, node in enumerate(leave_nodes)}
        else:
            leaf_to_hidden = {node: leaves_hidden[i, :] for i, node in enumerate(leave_nodes)}

        root_hidden, root_tag, all_tags = self.forward_node(root, leaf_to_hidden)

        # all_tags is a list (L length) of tag_size vectors
        return torch.stack(all_tags)

class BPModel(nn.Module):
    def __init__(self, vocab_size, tag_size, hidden_size=128):
        super(BPModel, self).__init__()
        self.vocab_size = vocab_size
        self.tag_size = tag_size

        self.encoder = Encoder(vocab_size, hidden_size=hidden_size).to(device)

        # From hidden output to unary factor (tag space)
        self.hidden_to_uni_fact = nn.Linear(hidden_size, tag_size)

        # For pairwise combination of hidden outputs, for intermediate nodes
        self.pair_to_hidden = nn.Linear(2 * hidden_size, hidden_size)

        # From edge-adjacent hidden output to binary factor (tag space^2)
        self.edge_potential = nn.Linear(2 * hidden_size, tag_size**2)

    def build_node_to_hidden(self, node, leaves_hidden):
        if node.is_leaf:
            return leaves_hidden[node], {node: leaves_hidden[node]}
        else:
            assert len(node.children) > 0
            child_left, child_right = node.children[0], node.children[-1]

            # Calculate children
            left_hidden, left_node_to_hidden = self.build_node_to_hidden(child_left, leaves_hidden)
            right_hidden, right_node_to_hidden = self.build_node_to_hidden(child_right, leaves_hidden)

            # Calculate current
            curr_hidden = self.pair_to_hidden(torch.cat([left_hidden, right_hidden], dim=0))

            # Combine maps
            results = {node: curr_hidden}
            results.update(left_node_to_hidden)
            results.update(right_node_to_hidden)

            return curr_hidden, results

    def build_factors(self, node, node_to_hidden):
        unary_factors = {}
        edge_factors = {}

        unary_factors[node] = self.hidden_to_uni_fact(node_to_hidden[node])

        if node.is_leaf:
            return unary_factors, edge_factors
        else:
            for child in node.children:
                # Calculate current
                edge_hidden = self.edge_potential(
                    torch.cat([
                        node_to_hidden[node],
                        node_to_hidden[child]], dim=0))
                edge_hidden = edge_hidden.reshape((self.tag_size, self.tag_size))

                edge_factors[(node, child)] = edge_hidden

                # Calculate children
                child_unary_fs, child_edge_fs = self.build_factors(child, node_to_hidden)

                # Combine results
                unary_factors.update(child_unary_fs)
                edge_factors.update(child_edge_fs)

            return unary_factors, edge_factors

    def messages_except(self, messages, exclude, expect_one=False):
        results = {node: message for node, message in messages.items() if node != exclude}
        if expect_one:
            assert(len(results) == 1)
            return list(results.values())[0]
        else:
            return list(results.values())

    def upwards_pass(self, root, unary_factors, edge_factors, variable_messages, factor_messages):
        postorder_nodes = tree_in_postorder(root)

        for node in postorder_nodes:
            # All nodes have an in-message from their unary factor
            unary_factor = unary_factors[node]
            variable_messages[node]["unary"] = unary_factor

            parent = node.parent
            if parent is not None:
                edge = (parent, node)

                # Variable to Factor
                # Find all children factors (1 or 2) of variable, and unary factor
                if node.is_leaf:
                    # No children, just propagate unary factor
                    node_to_factor_msg = unary_factor
                else:
                    left_child, right_child = node.children[0], node.children[-1]
                    left_edge, right_edge = (node, left_child), (node, right_child)
                    left_msg = self.messages_except(factor_messages[left_edge], node, expect_one=True)
                    right_msg = self.messages_except(factor_messages[right_edge], node, expect_one=True)

                    # Element-wise multiplication (log-space add) across children
                    if (left_child == right_child):
                        # If only had 1 child
                        node_to_factor_msg = left_msg + unary_factor
                    else:
                        node_to_factor_msg = left_msg + right_msg + unary_factor

                # assert node_to_factor_msg.shape == (self.tag_size, self.tag_size)
                assert node_to_factor_msg.shape == (self.tag_size, )
                factor_messages[edge][node] = node_to_factor_msg

                # Factor to Variable
                # Since edge potential, only need message from bottom (node_to_factor_msg)
                edge_factor = edge_factors[edge]
                # Edge factor is (parent, child)
                # Make sure log-addition broadcasts across rows
                # logsumexp out the child axis
                factor_to_parent_msg = torch.logsumexp(edge_factor + node_to_factor_msg, dim=1)
                assert factor_to_parent_msg.shape == (self.tag_size, )
                variable_messages[parent][edge] = factor_to_parent_msg

        return variable_messages, factor_messages

    def downwards_pass(self, root, unary_factors, edge_factors, variable_messages, factor_messages):
        preorder_nodes = tree_in_preorder(root)

        for node in preorder_nodes:
            unary_factor = unary_factors[node]

            # Message from top edge into variable, excluding unary
            # Root node, only has unary factor
            if node.parent is None:
                pass
            else:
                top_edge = (node.parent, node)
                top_edge_factor = edge_factors[top_edge]
                # From parent to factor
                parent_msg = self.messages_except(factor_messages[top_edge], node, expect_one=True)
                # From factor to node
                # Edge factor is (parent, child)
                # Make sure log-addition broadcasts across cols
                # logsumexp out the parent axis
                factor_to_node_msg = torch.logsumexp(top_edge_factor + parent_msg.reshape((self.tag_size, 1)), dim=0)
                assert factor_to_node_msg.shape == (self.tag_size, )
                variable_messages[node][top_edge] = factor_to_node_msg

            # Message from variable to each child factor
            if node.is_leaf:
                # Update unary potential messages? But does not matter for this case
                pass
            else:
                for child in node.children:
                    child_edge = (node, child)
                    node_to_child_msgs = self.messages_except(variable_messages[node], child)
                    factor_messages[child_edge][node] = sum(node_to_child_msgs)  # Log-sum (product)

        return variable_messages, factor_messages


    def get_variable_beliefs(self, variable_messages):
        return {node: sum(messages.values()) for node, messages in variable_messages.items()}


    def forward(self, root):
        leave_nodes = tree_get_leaves(root)

        # Leaves input is vector of L
        leaves_input = torch.tensor(
            [word_dict[node.word] for node in leave_nodes],
            device=device)

        # Leaves hidden is L x hidden_size
        # But sometimes there is only 1 leaf

        leaves_hidden = self.encoder(leaves_input)
        if len(leave_nodes) == 1:
            assert len(leaves_hidden.shape) == 1
            leaf_to_hidden = {node: leaves_hidden for i, node in enumerate(leave_nodes)}
        else:
            leaf_to_hidden = {node: leaves_hidden[i, :] for i, node in enumerate(leave_nodes)}

        # Build factors
        root_hidden, node_to_hidden = self.build_node_to_hidden(root, leaf_to_hidden)
        unary_factors, edge_factors = self.build_factors(root, node_to_hidden)

        # Build messages
        def init_messages():
            return {}
        variable_messages = defaultdict(init_messages)
        factor_messages = defaultdict(init_messages)

        # Belief propagation
        variable_messages, factor_messages = self.upwards_pass(
            root, unary_factors, edge_factors, variable_messages, factor_messages)

        # Sanity Check
        assert all(
                len(msgs) == len(node.children) + 1
                for node, msgs in variable_messages.items())
        assert all(
                len(msgs) == 1
                for edge, msgs in factor_messages.items())

        variable_messages, factor_messages = self.downwards_pass(
            root, unary_factors, edge_factors, variable_messages, factor_messages)

        # Sanity Check
        assert all(
                len(msgs) == len(node.children) + 1 + (0 if node.parent is None else 1)
                for node, msgs in variable_messages.items())
        assert all(
                len(msgs) == 2
                for edge, msgs in factor_messages.items())

        node_beliefs = self.get_variable_beliefs(variable_messages)

        # Order beliefs
        postorder_nodes = tree_in_postorder(root)
        beliefs = torch.stack([node_beliefs[node] for node in postorder_nodes])

        # Dont need to normalize beliefs here,
        # as the softmax layer in CrossEntropyLoss normalizes the log scores
        return beliefs

def train(model, train_trees, test_trees, n_epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = nn.CrossEntropyLoss()

    train_losses = []
    train_accs = []
    train_leaf_accs = []
    test_losses = []
    test_accs = []
    test_leaf_accs = []

    for epoch in range(n_epochs):

        train_epoch_loss = 0.0
        train_epoch_acc = 0.0
        train_epoch_leaf_acc = 0.0
        test_epoch_loss = 0.0
        test_epoch_acc = 0.0
        test_epoch_leaf_acc = 0.0

        # Each "batch" is just 1 sentence
        np.random.shuffle(train_trees)
        for tree in train_trees:
            optimizer.zero_grad()

            # Generate predictions in postorder traversal
            pred_tags = model(tree)

            # Generate actuals in postorder traversal
            nodes = tree_in_postorder(tree)
            actual_tags = torch.tensor(
                [node.true_label for node in nodes],
                device=device)
            is_leaf = torch.tensor(
                [node.is_leaf for node in nodes],
                device=device)

            loss = loss_fn(pred_tags, actual_tags)
            loss.backward()
            optimizer.step()

            train_epoch_loss += loss.item()
            tree_acc = accuracy(torch.argmax(pred_tags, dim=1), actual_tags)
            leaf_acc = accuracy(torch.argmax(pred_tags, dim=1)[is_leaf], actual_tags[is_leaf])
            train_epoch_acc += tree_acc
            train_epoch_leaf_acc += leaf_acc

        for tree in test_trees:
            pred_tags = model(tree)

            nodes = tree_in_postorder(tree)
            actual_tags = torch.tensor(
                [node.true_label for node in nodes],
                device=device)
            is_leaf = torch.tensor(
                [node.is_leaf for node in nodes],
                device=device)

            loss = loss_fn(pred_tags, actual_tags)
            test_epoch_loss += loss.item()
            tree_acc = accuracy(torch.argmax(pred_tags, dim=1), actual_tags)
            leaf_acc = accuracy(torch.argmax(pred_tags, dim=1)[is_leaf], actual_tags[is_leaf])
            test_epoch_acc += tree_acc
            test_epoch_leaf_acc += leaf_acc

        train_epoch_loss /= len(train_trees)
        train_epoch_acc /= len(train_trees)
        train_epoch_leaf_acc /= len(train_trees)
        test_epoch_loss /= len(test_trees)
        test_epoch_acc /= len(test_trees)
        test_epoch_leaf_acc /= len(test_trees)

        print(f"({epoch})\t Train >> AC: {train_epoch_acc:.4f}, LAC: {train_epoch_leaf_acc:.4f}, CE: {train_epoch_loss:.4f}")
        print(f"\t Test >> AC: {test_epoch_acc:.4f}, LAC: {test_epoch_leaf_acc:.4f}, CE: {test_epoch_loss:.4f}")

        train_losses.append(train_epoch_loss)
        train_accs.append(train_epoch_acc)
        train_leaf_accs.append(train_epoch_leaf_acc)
        test_losses.append(test_epoch_loss)
        test_accs.append(test_epoch_acc)
        test_leaf_accs.append(test_epoch_leaf_acc)

    return {
        "train": {"loss": train_losses, "ac": train_accs, "lac": train_leaf_accs},
        "test": {"loss": test_losses, "ac": test_accs, "lac": test_leaf_accs}
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Consistuency Parsing")
    parser.add_argument("--mode", choices=["train", "plot"], default="train")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--bpresults")
    parser.add_argument("--baselineresults")
    parser.add_argument("--method", choices=["baseline", "bp"], default="bp")
    parser.add_argument("--data", default="ptb-munged.mrg.bin.oneline.txt")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "train":
        # Load data
        print("Loading raw data...")
        with open(args.data) as f:
            lines = [l.strip() for l in f.readlines()]
        print("Parsing tree strings...")
        trees = [Tree.fromstring(tr1) for tr1 in lines]
        print("Building trees...")
        my_trees = [build_tree(tree) for tree in trees]

        # Shuffle and Split Data
        n = len(my_trees)
        train_n = int(0.7 * n)
        test_n = n - train_n
        np.random.shuffle(my_trees)
        train_trees = my_trees[:train_n]
        test_trees = my_trees[train_n:]

        print(f"Using device {device}")
        print(f"Starting training in {args.method} method on {n} rows...")
        if args.method == "baseline":
            model = Baseline(len(word_dict), len(tag_dict)).to(device)
        elif args.method == "bp":
            model = BPModel(len(word_dict), len(tag_dict)).to(device)
        else:
            print(f"Invalid method specified '{args.method}'")
            exit()
        results = train(model, train_trees, test_trees, args.epochs)

        # Done with training
        fmt = "%Y%m%d-%H%M"
        results_filename = f"results_{args.method}_{dt.datetime.now().strftime(fmt)}.json"
        print(f"Writing results to {results_filename}")
        with open(results_filename, "w") as f:
            json.dump(results, f)

    elif args.mode == "plot":
        if args.bpresults is None or args.baselineresults is None:
            print("No results file specified! Use --bpresults and --baselineresults")
            exit()

        with open(args.bpresults) as f:
            bp_results = json.load(f)
        with open(args.baselineresults) as f:
            bl_results = json.load(f)

        iters = range(len(bl_results["train"]["loss"]))

        plt.figure(figsize=(8, 6))
        plt.plot(iters, bl_results["train"]["loss"], label="Baseline Train", linestyle="--", c="C0")
        plt.plot(iters, bl_results["test"]["loss"], label="Baseline Test", c="C0")
        plt.plot(iters, bp_results["train"]["loss"], label="Belief Prop. Train", linestyle="--", c="C1")
        plt.plot(iters, bp_results["test"]["loss"], label="Belief Prop. Test", c="C1")
        plt.xlabel("Iterations")
        plt.ylabel("Cross Entropy")
        plt.title("Cross Entropy over Iterations")
        plt.legend()
        plt.savefig("crossentropy")
