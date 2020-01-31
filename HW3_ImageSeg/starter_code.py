import os
import pdb
from tqdm import tqdm

import matplotlib.pyplot as plt

from PIL import Image
import torchvision.transforms.functional as TF

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ortools.linear_solver import pywraplp

NUM_CLASSES = 2
NUM_FEATURES = 100

train_file = "./train.txt"
test_file = "./test.txt"

# train_file = "./train_small.txt"
# test_file = "./test_small.txt"

class VOCDataset(Dataset):
    """Class to store VOC semantic segmentation dataset"""

    def __init__(self, image_dir, label_dir, file_list):

        self.image_dir = image_dir
        self.label_dir = label_dir
        reader = open(file_list, "r")
        self.files = []
        for file in reader:
            self.files.append(file.strip())

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        # 0 stands for background, 1 for foreground
        labels = np.load(os.path.join(self.label_dir, fname+".npy"))
        labels[labels > 0.0] = 1.0
        image = Image.open(os.path.join(self.image_dir, fname+".jpg"), "r")
        sample = (TF.to_tensor(image), torch.LongTensor(labels))

        return sample


class AlexNet(nn.Module):
    """Class defining AlexNet layers used for the convolutional network"""

    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=2, padding=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return x


class FCNHead(nn.Sequential):
    """Class defining FCN (fully convolutional network) layers"""

    def __init__(self, in_channels, channels):
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        ]

        super(FCNHead, self).__init__(*layers)


class SimpleSegmentationModel(nn.Module):
    """
    Class defining end-to-end semantic segmentation model.
    It combines AlexNet and FCN layers with interpolation for deconvolution.
    This model is pretrained using cross-entropy loss.
    After pre-training, use the get_repr() function to construct 32x32x100 feature tensors for each image
    """

    def __init__(self, n_feat, n_classes):
        super(SimpleSegmentationModel, self).__init__()
        self.n_feat = n_feat
        self.backbone = AlexNet()
        self.classifier = FCNHead(256, n_feat)
        self.linear = nn.Linear(n_feat, n_classes)

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        x = self.classifier(features)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        x = x.permute(0, 2, 3, 1)
        x = x.contiguous().view(-1, self.n_feat)
        x = self.linear(x)

        return x

    def get_repr(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        x = self.classifier(features)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        x = x.permute(0, 2, 3, 1)
        return x


class LinearSVM(nn.Module):

    def __init__(self, n_feat, n_classes):
        super(LinearSVM, self).__init__()
        self.n_feat = n_feat
        self.n_classes = n_classes
        # TODO: Define weights for linear SVM
        self.linear = nn.Linear(n_feat, n_classes)

    def forward(self, x):
        # TODO: Define forward function for linear SVM
        x = x.contiguous().view(-1, self.n_feat)
        return self.linear(x)


class StructSVM(nn.Module):

    def __init__(self, n_feat, n_classes, w, h):
        super(StructSVM, self).__init__()
        self.n_feat = n_feat
        self.n_classes = n_classes
        self.w = w
        self.h = h

        edges = self.generate_edge_indices()
        self.edges = np.array(edges)
        assert (self.edges.shape ==
                (2 * (self.w - 1) * (self.h - 1) + (self.w - 1) + (self.h - 1), 2))
        # Reverse lookup
        self.edges2idx = {(u, v): i for i, (u, v) in enumerate(self.edges)}


        # TODO: Define weights for structured SVM
        self.pixel_w = nn.Linear(self.n_feat, self.n_classes)
        self.edge_w = nn.Linear(self.n_feat * 2, self.n_classes)

    def generate_edge_indices(self):
        edge_idx = []
        dim = (self.w, self.h)
        for i in range(self.w):
            for j in range(self.h):
                u = np.ravel_multi_index((i, j), dims=dim)
                if i + 1 < self.w:
                    v = np.ravel_multi_index((i+1, j), dims=dim)
                    edge_idx.append((u, v))
                if j + 1 < self.h:
                    v = np.ravel_multi_index((i, j+1), dims=dim)
                    edge_idx.append((u, v))
        return edge_idx

    def forward(self, image):
        # TODO: Define forward function for structured SVM
        x = image.contiguous().view(-1, self.n_feat)

        # Calculate pixel potentials
        pixel_pots = self.pixel_w(x)
        assert (pixel_pots.shape ==
                (self.w * self.h, self.n_classes))

        # Calculate concatenated features of edges
        edge_feats = torch.cat((x[self.edges[:, 0]], x[self.edges[:, 1]]), dim=1)
        assert (edge_feats.shape ==
                (len(self.edges), 2 * self.n_feat))

        # Calculate edge potentials
        edge_pots = self.edge_w(edge_feats)
        assert (edge_pots.shape ==
                (len(self.edges), self.n_classes))

        return pixel_pots, edge_pots

    def predict(self, pixel_pots, edge_pots, true_assignment=None):
        solver = pywraplp.Solver("LinearExample", pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)

        # Variables
        pixel_assignments = {}
        edge_assignments = {}
        for i in range(self.w * self.h):
            for c in range(self.n_classes):
                pixel_assignments[i, c] = solver.NumVar(0.0, 1.0, f"x[{i}, {c}]")
        for i, j in self.edges:
            for c in range(self.n_classes):
                edge_assignments[i, j, c] = solver.NumVar(0.0, 1.0, f"y[{i}, {j}, {c}]")

        # Objective
        objective = 0.0
        for i in range(self.w * self.h):
            for c in range(self.n_classes):
                objective += pixel_pots[i, c] * pixel_assignments[i, c]
        for i, j in self.edges:
            for c in range(self.n_classes):
                objective += edge_pots[self.edges2idx[i, j], c] * edge_assignments[i, j, c]


        # Augmented Loss Objective
        if true_assignment is not None:
            assert true_assignment.shape == (self.w * self.h, self.n_classes)
            hamming_loss = 0.0
            for i in range(self.w * self.h):
                for c in range(self.n_classes):
                    hamming_loss += true_assignment[i, c] * (1 - pixel_assignments[i, c]) + \
                                    pixel_assignments[i, c] * (1 - true_assignment[i, c])
            hamming_loss /= self.w * self.h * self.n_classes
            objective += hamming_loss

        solver.Maximize(objective)

        # Constraints
        for i in range(self.w * self.h):
            solver.Add(solver.Sum([pixel_assignments[i, c] for c in range(self.n_classes)]) == 1)
        for i, j in self.edges:
            for c in range(self.n_classes):
                solver.Add(edge_assignments[i, j, c] <= pixel_assignments[i, c])
                solver.Add(edge_assignments[i, j, c] <= pixel_assignments[j, c])

        result = solver.Solve()
        assert result == pywraplp.Solver.OPTIMAL
        assert solver.VerifySolution(1e-7, True)

        # Write results into matrices
        final_pixel_ass = np.zeros((self.w * self.h, self.n_classes))
        final_edge_ass = np.zeros((len(self.edges), self.n_classes))

        for i in range(self.w * self.h):
            for c in range(self.n_classes):
                final_pixel_ass[i, c] = pixel_assignments[i, c].solution_value()

        for i, j in self.edges:
            for c in range(self.n_classes):
                final_edge_ass[self.edges2idx[i, j], c] = edge_assignments[i, j, c].solution_value()

        return np.round(final_pixel_ass), np.round(final_edge_ass)

    def score(self, pixel_pots, pixel_assignments, edge_pots, edge_assignments):
        assert pixel_pots.shape == (self.w * self.h, self.n_classes)
        assert pixel_assignments.shape == (self.w * self.h, self.n_classes)
        assert edge_pots.shape == (len(self.edges), self.n_classes)
        assert edge_assignments.shape == (len(self.edges), self.n_classes)

        s = torch.sum(pixel_pots * torch.tensor(pixel_assignments)) + \
            torch.sum(edge_pots * torch.tensor(edge_assignments))

        return s

def train_cnn(model, train_batches, test_batches, num_epochs):
    """
    This function runs a training loop for the FCN semantic segmentation model
    """
    criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1, 4]))
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0.0
        total = 0.0
        class_gold = [0.0] * NUM_CLASSES
        class_pred = [0.0] * NUM_CLASSES
        class_correct = [0.0] * NUM_CLASSES
        for i, batch in enumerate(tqdm(train_batches)):
            optimizer.zero_grad()
            images, labels = batch
            output = model(images)
            labels = labels.contiguous().view(-1, 1).squeeze()
            loss = criterion(output, labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            # Metrics
            _, output = torch.max(output, axis=1)
            output = output.squeeze().detach().numpy()
            labels = labels.contiguous().view(-1, 1).squeeze().numpy()
            cur_class_pred = np.unique(output, return_counts=True)
            for key, val in zip(cur_class_pred[0], cur_class_pred[1]):
                class_pred[key] += val
            cur_class_gold = np.unique(labels, return_counts=True)
            for key, val in zip(cur_class_gold[0], cur_class_gold[1]):
                class_gold[key] += val
            cur_correct = (output == labels).tolist()
            for j, val in enumerate(cur_correct):
                if val:
                    class_correct[labels[j]] += 1
            correct += np.sum(cur_correct)
            total += len(labels)

        class_iou = [x/(y+z-x) for x, y, z in zip(class_correct, class_gold, class_pred)]
        mean_iou = sum(class_iou) / len(class_correct)
        print("\tEpoch {} Training Loss: {}".format(epoch, total_loss/len(train_batches)))
        print("\tEpoch {} Mean IOU: {}".format(epoch, mean_iou))
        print("\tEpoch {} Pixel Accuracy: {}".format(epoch, correct / total))
        test_cnn(model, test_batches)


def test_cnn(model, test_batches):
    """
        This function evaluates the FCN semantic segmentation model on the test set
    """
    criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1, 4]))
    total_loss = 0.0
    correct = 0.0
    total = 0.0
    class_gold = [0.0] * NUM_CLASSES
    class_pred = [0.0] * NUM_CLASSES
    class_correct = [0.0] * NUM_CLASSES
    for i, batch in enumerate(tqdm(test_batches)):
        images, labels = batch
        output = model(images)

        loss = criterion(output, labels.contiguous().view(-1, 1).squeeze())
        total_loss += loss.detach().item()

        _, output = torch.max(output, axis=1)
        # visualize_grayscale_image(output.view(32, 32).detach().numpy(), i)
        output = output.squeeze().detach().numpy()
        labels = labels.contiguous().view(-1, 1).squeeze().numpy()
        cur_class_pred = np.unique(output, return_counts=True)
        for key, val in zip(cur_class_pred[0], cur_class_pred[1]):
            class_pred[key] += val
        cur_class_gold = np.unique(labels, return_counts=True)
        for key, val in zip(cur_class_gold[0], cur_class_gold[1]):
            class_gold[key] += val
        cur_correct = (output == labels).tolist()
        for j, val in enumerate(cur_correct):
            if val:
                class_correct[labels[j]] += 1
        correct += np.sum(cur_correct)
        total += len(labels)
    class_iou = [x/(y+z-x) for x, y, z in zip(class_correct, class_gold, class_pred)]
    mean_iou = sum(class_iou) / len(class_correct)
    print("\tTest Loss: {}".format(total_loss / len(test_batches)))
    print("\tMean IOU: {}".format(mean_iou))
    print("\tPixel Accuracy: {}".format(correct / total))


def train_linear_svm(cnn_model, svm_model, train_batches, test_batches, num_epochs):
    # TODO: Write a training loop for the linear SVM
    # Keep in mind that the CNN model is needed to compute features, but it should not be finetuned
    criterion = nn.MultiMarginLoss(weight=torch.Tensor([1, 4]))
    optimizer = optim.Adam(svm_model.parameters(), lr=0.0001)
    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0.0
        total = 0.0
        class_gold = [0.0] * NUM_CLASSES
        class_pred = [0.0] * NUM_CLASSES
        class_correct = [0.0] * NUM_CLASSES

        print(f"Training linear SVM epoch {epoch}")
        for i, batch in enumerate(tqdm(train_batches)):
            optimizer.zero_grad()
            images, labels = batch
            feats = cnn_model.get_repr(images)
            output = svm_model(feats)
            labels = labels.contiguous().view(-1, 1).squeeze()
            loss = criterion(output, labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            # Metrics
            _, output = torch.max(output, axis=1)
            output = output.squeeze().detach().numpy()
            labels = labels.contiguous().view(-1, 1).squeeze().numpy()
            cur_class_pred = np.unique(output, return_counts=True)
            for key, val in zip(cur_class_pred[0], cur_class_pred[1]):
                class_pred[key] += val
            cur_class_gold = np.unique(labels, return_counts=True)
            for key, val in zip(cur_class_gold[0], cur_class_gold[1]):
                class_gold[key] += val
            cur_correct = (output == labels).tolist()
            for j, val in enumerate(cur_correct):
                if val:
                    class_correct[labels[j]] += 1
            correct += np.sum(cur_correct)
            total += len(labels)

        class_iou = [x/(y+z-x) for x, y, z in zip(class_correct, class_gold, class_pred)]
        mean_iou = sum(class_iou) / len(class_correct)
        print("\tEpoch {} Training Loss: {}".format(epoch, total_loss/len(train_batches)))
        print("\tEpoch {} Mean IOU: {}".format(epoch, mean_iou))
        print("\tEpoch {} Pixel Accuracy: {}".format(epoch, correct / total))
        test_linear_svm(cnn_model, svm_model, test_batches)


def test_linear_svm(cnn_model, svm_model, test_batches):
    # TODO: Write a testing function for the linear SVM
    criterion = nn.MultiMarginLoss(weight=torch.Tensor([1, 4]))
    total_loss = 0.0
    correct = 0.0
    total = 0.0
    class_gold = [0.0] * NUM_CLASSES
    class_pred = [0.0] * NUM_CLASSES
    class_correct = [0.0] * NUM_CLASSES
    for i, batch in enumerate(tqdm(test_batches)):
        images, labels = batch
        feats = cnn_model.get_repr(images)
        output = svm_model(feats)

        loss = criterion(output, labels.contiguous().view(-1, 1).squeeze())
        total_loss += loss.detach().item()

        _, output = torch.max(output, axis=1)
        # visualize_grayscale_image(output.view(32, 32).detach().numpy(), i)
        output = output.squeeze().detach().numpy()
        labels = labels.contiguous().view(-1, 1).squeeze().numpy()
        cur_class_pred = np.unique(output, return_counts=True)
        for key, val in zip(cur_class_pred[0], cur_class_pred[1]):
            class_pred[key] += val
        cur_class_gold = np.unique(labels, return_counts=True)
        for key, val in zip(cur_class_gold[0], cur_class_gold[1]):
            class_gold[key] += val
        cur_correct = (output == labels).tolist()
        for j, val in enumerate(cur_correct):
            if val:
                class_correct[labels[j]] += 1
        correct += np.sum(cur_correct)
        total += len(labels)
    class_iou = [x/(y+z-x) for x, y, z in zip(class_correct, class_gold, class_pred)]
    mean_iou = sum(class_iou) / len(class_correct)
    print("\tTest Loss: {}".format(total_loss / len(test_batches)))
    print("\tMean IOU: {}".format(mean_iou))
    print("\tPixel Accuracy: {}".format(correct / total))

# TODO: Write a function to compute the structured hinge loss
# using the max-scoring output from the ILP and the gold output
def compute_struct_svm_loss(svm_model,
                            pixel_pots, pixel_assignments, true_pixel_assignments,
                            edge_pots, edge_assignments, true_edge_assignments):

    assert pixel_assignments.shape == true_pixel_assignments.shape
    assert edge_assignments.shape == true_edge_assignments.shape
    assert pixel_assignments.shape == (svm_model.w * svm_model.h, svm_model.n_classes)
    assert edge_assignments.shape == (len(svm_model.edges), svm_model.n_classes)

    # Calculate terms for Structured Hinge Loss
    score = svm_model.score(pixel_pots, pixel_assignments, edge_pots, edge_assignments)
    true_score = svm_model.score(pixel_pots, true_pixel_assignments, edge_pots, true_edge_assignments)
    hamming_loss = (pixel_assignments != true_pixel_assignments).sum()
    hamming_loss /= svm_model.w * svm_model.h * svm_model.n_classes

    # Structured Hinge Loss
    loss, _ = torch.max(score + hamming_loss - true_score, 0)

    return loss


def train_struct_svm(cnn_model, svm_model, train_batches, test_batches, num_epochs):
    # TODO: Write a training loop for the structured SVM
    # Keep in mind that the CNN model is needed to compute features, but it should not be finetuned
    optimizer = optim.Adam(svm_model.parameters(), lr=0.0001)
    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0.0
        total = 0.0
        class_gold = [0.0] * NUM_CLASSES
        class_pred = [0.0] * NUM_CLASSES
        class_correct = [0.0] * NUM_CLASSES

        print(f"Training struct SVM epoch {epoch}")
        for i, batch in enumerate(tqdm(train_batches)):
            optimizer.zero_grad()
            images, labels = batch
            labels = labels.contiguous().view(-1, 1).squeeze()

            # Format true assignments
            true_pixel_assignments = np.zeros((svm_model.w * svm_model.h, svm_model.n_classes))
            true_pixel_assignments[np.arange(svm_model.w * svm_model.h), labels] = 1
            assert np.all(labels.numpy() == np.argmax(true_pixel_assignments, axis=1))
            true_edge_assignments = np.zeros((len(svm_model.edges), svm_model.n_classes))
            for i, j in svm_model.edges:
                if labels[i] == labels[j]:
                    c = labels[i]
                    true_edge_assignments[svm_model.edges2idx[i, j], c] = 1

            feats = cnn_model.get_repr(images)
            pixel_pots, edge_pots = svm_model.forward(feats)

            # Get loss-augmented MAP assignment via ILP
            pixel_assignments, edge_assignments = svm_model.predict(
                    pixel_pots.detach().numpy(),
                    edge_pots.detach().numpy(),
                    true_assignment=true_pixel_assignments)
            output = np.argmax(pixel_assignments, axis=1)

            # Calculate structured hinge loss
            loss = compute_struct_svm_loss(
                    svm_model,
                    pixel_pots, pixel_assignments, true_pixel_assignments,
                    edge_pots, edge_assignments, true_edge_assignments)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            # Metrics
            labels = labels.contiguous().view(-1, 1).squeeze().numpy()
            cur_class_pred = np.unique(output, return_counts=True)
            for key, val in zip(cur_class_pred[0], cur_class_pred[1]):
                class_pred[key] += val
            cur_class_gold = np.unique(labels, return_counts=True)
            for key, val in zip(cur_class_gold[0], cur_class_gold[1]):
                class_gold[key] += val
            cur_correct = (output == labels).tolist()
            for j, val in enumerate(cur_correct):
                if val:
                    class_correct[labels[j]] += 1
            correct += np.sum(cur_correct)
            total += len(labels)

        class_iou = [x/(y+z-x) for x, y, z in zip(class_correct, class_gold, class_pred)]
        mean_iou = sum(class_iou) / len(class_correct)
        print("\tEpoch {} Training Loss: {}".format(epoch, total_loss/len(train_batches)))
        print("\tEpoch {} Mean IOU: {}".format(epoch, mean_iou))
        print("\tEpoch {} Pixel Accuracy: {}".format(epoch, correct / total))
        test_struct_svm(cnn_model, svm_model, test_batches)


def test_struct_svm(cnn_model, svm_model, test_batches):
    # TODO: Write a testing function for the structured SVM
    total_loss = 0.0
    correct = 0.0
    total = 0.0
    class_gold = [0.0] * NUM_CLASSES
    class_pred = [0.0] * NUM_CLASSES
    class_correct = [0.0] * NUM_CLASSES
    for i, batch in enumerate(tqdm(test_batches)):
        images, labels = batch
        feats = cnn_model.get_repr(images)
        pixel_pots, edge_pots = svm_model.forward(feats)

        # Get regular MAP assignment via ILP
        pixel_assignments, edge_assignments = svm_model.predict(
                pixel_pots.detach().numpy(),
                edge_pots.detach().numpy())
        assert np.all(pixel_assignments.sum(axis=1) == 1.0)
        output = np.argmax(pixel_assignments, axis=1)

        labels = labels.contiguous().view(-1, 1).squeeze().numpy()

        visualize_grayscale_image(labels.reshape(32, 32), f"segments/struct_svm/{i}_actual")
        visualize_grayscale_image(output.reshape(32, 32), f"segments/struct_svm/{i}_predicted")

        true_pixel_assignments = np.zeros((svm_model.w * svm_model.h, svm_model.n_classes))
        true_pixel_assignments[np.arange(svm_model.w * svm_model.h), labels] = 1
        assert np.all(labels.numpy() == np.argmax(true_pixel_assignments, axis=1))
        true_edge_assignments = np.zeros((len(svm_model.edges), svm_model.n_classes))
        for i, j in svm_model.edges:
            if labels[i] == labels[j]:
                c = labels[i]
                true_edge_assignments[svm_model.edges2idx[i, j], c] = 1

        loss = compute_struct_svm_loss(
                svm_model,
                pixel_pots, pixel_assignments, true_pixel_assignments,
                edge_pots, edge_assignments, true_edge_assignments)

        total_loss += loss.detach().item()

        cur_class_pred = np.unique(output, return_counts=True)
        for key, val in zip(cur_class_pred[0], cur_class_pred[1]):
            class_pred[key] += val
        cur_class_gold = np.unique(labels, return_counts=True)
        for key, val in zip(cur_class_gold[0], cur_class_gold[1]):
            class_gold[key] += val
        cur_correct = (output == labels).tolist()
        for j, val in enumerate(cur_correct):
            if val:
                class_correct[labels[j]] += 1
        correct += np.sum(cur_correct)
        total += len(labels)
    class_iou = [x/(y+z-x) for x, y, z in zip(class_correct, class_gold, class_pred)]
    mean_iou = sum(class_iou) / len(class_correct)
    print("\tTest Loss: {}".format(total_loss / len(test_batches)))
    print("\tMean IOU: {}".format(mean_iou))
    print("\tPixel Accuracy: {}".format(correct / total))

def visualize_grayscale_image(image, file=None):
    plt.imshow(image, cmap="gray")
    # Uncomment this to visualize image
    # plt.show()
    # Uncomment this to save image
    plt.savefig(str(file)+".png")


if __name__ == "__main__":

    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Uncomment following lines after providing appropriate paths
    train_dataset = VOCDataset("./DownsampledImages/", "./DownsampledLabels/", train_file)
    test_dataset = VOCDataset("./DownsampledImages/", "./DownsampledLabels/", test_file)

    train_batches = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_batches = DataLoader(test_dataset, batch_size=1, shuffle=True)

    cnn = SimpleSegmentationModel(NUM_FEATURES, NUM_CLASSES)
    linear_svm = LinearSVM(NUM_FEATURES, NUM_CLASSES)
    struct_svm = StructSVM(NUM_FEATURES, NUM_CLASSES, 32, 32)

    cnn_model_file = "cnn_model.pt"
    if os.path.exists(cnn_model_file):
        print("Found pre-trained CNN")
        cnn.load_state_dict(torch.load(cnn_model_file))
        print("Loaded pre-trained CNN")
    else:
        print("Did not find pre-trained CNN")
        train_cnn(cnn, train_batches, test_batches, 2)

        print("Done pre-training CNN, saving...")
        torch.save(cnn.state_dict(), "cnn_model.pt")

    test_cnn(cnn, test_batches)

    # TODO: Instantiate a linear SVM and call train/ test functions
    if os.path.exists("linear_svm_model.pt"):
        linear_svm.load_state_dict(torch.load("linear_svm_model.pt"))
    else:
        train_linear_svm(cnn, linear_svm, train_batches, test_batches, 3)
        torch.save(linear_svm.state_dict(), "linear_svm_model.pt")
        test_linear_svm(cnn, linear_svm, test_batches)

    # TODO: Instantiate a structured SVM and call train/ test functions
    if os.path.exists("struct_svm_model.pt"):
        struct_svm.load_state_dict(torch.load("struct_svm_model.pt"))
    else:
        train_struct_svm(cnn, struct_svm, train_batches, test_batches, 3)
        torch.save(struct_svm.state_dict(), "struct_svm_model.pt")
        test_struct_svm(cnn, struct_svm, test_batches)

    # Visualize 3 images
    vis_batches = enumerate(test_batches)
    for i in range(3):
        _, (vis_img, vis_label) = next(vis_batches)

        # Actual
        visualize_grayscale_image(vis_label.squeeze().numpy(),
                                  f"actual_{i}")

        # FCN
        output = cnn(vis_img)
        _, output = torch.max(output, axis=1)
        visualize_grayscale_image(output.view(32, 32).detach().numpy(),
                                  f"fcn_predicted_{i}")

        # Linear svm
        feats = cnn.get_repr(vis_img)
        output = linear_svm(feats)
        _, output = torch.max(output, axis=1)
        visualize_grayscale_image(output.view(32, 32).detach().numpy(),
                                  f"linear_svm_predicted_{i}")

        # Struct SVM
        feats = cnn.get_repr(vis_img)
        pixel_pots, edge_pots = struct_svm.forward(feats)
        # Get regular MAP assignment via ILP
        pixel_assignments, _ = struct_svm.predict(
                pixel_pots.detach().numpy(),
                edge_pots.detach().numpy())
        output = np.argmax(pixel_assignments, axis=1)
        visualize_grayscale_image(output.reshape(32, 32),
                                  f"struct_svm_predicted_{i}")
