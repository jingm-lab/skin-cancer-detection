# %%
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
)


# %%
class ClassificationModelBuilding:
    def __init__(
        self,
        model,
        loss_fn,
        optimizer,
        train_loader,
        val_loader=None,
        scheduler=None,
        early_stopping_patience=None,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.early_stopping_patience = early_stopping_patience
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        self.train_loss = []
        self.val_loss = []

        self.train_acc = []
        self.val_acc = []

        self.train_recall = []
        self.val_recall = []

        self.train_f1 = []
        self.val_f1 = []

        self.train_precision = []
        self.val_precision = []

        self.train_auc = []
        self.val_auc = []

        self.total_epochs = 0

        if early_stopping_patience is not None:
            self.best_val_loss = float("inf")
            self.early_stopping_counter = 0
            # Store model's state_dict for the best epoch
            self.best_model_state = None

    def to(self, device):
        try:
            self.device = device
            self.model.to(self.device)
        except RuntimeError:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Couldn't send to {device}. Sending to {self.device} instead.")
            self.model.to(self.device)

    def set_seed(self, seed=42):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        try:
            self.train_loader.sampler.generator.manual_seed(seed)
        except AttributeError:
            pass

    def train_step(self, X, y):
        self.model.train()
        X = X.to(self.device)
        y = y.to(self.device).float().unsqueeze(1)
        yhat = self.model(X)

        # Inception V3 returns a tuple (logits, aux_logits)
        if isinstance(yhat, tuple):
            yhat = yhat[0]

        loss = self.loss_fn(yhat, y)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()

    def validate_step(self, X, y):
        self.model.eval()
        X = X.to(self.device)
        y = y.to(self.device).float().unsqueeze(1)
        yhat = self.model(X)

        if isinstance(yhat, tuple):
            yhat = yhat[0]

        loss = self.loss_fn(yhat, y)
        return loss.item()

    def _mini_batch(self, validation=False):
        if validation:
            data_loader = self.val_loader
            step_fn = self.validate_step
        else:
            data_loader = self.train_loader
            step_fn = self.train_step

        if data_loader is None:
            return None

        mini_batch_loss = []
        all_preds = []
        all_ys = []

        for x_batch, y_batch in data_loader:
            # Move y_batch to cpu for performance metrics calculation
            y_batch_cpu = y_batch.cpu()

            loss = step_fn(x_batch, y_batch)
            mini_batch_loss.append(loss)

            pred = self.predict(x_batch)
            all_preds.append(pred)
            all_ys.append(y_batch_cpu)

        # Concatenate all bathces
        all_preds = torch.cat(all_preds).numpy()
        all_ys = torch.cat(all_ys).numpy()
        loss = np.mean(mini_batch_loss)

        return loss, all_preds, all_ys

    def _compute_metrics(self, y_true, y_pred_prob):
        y_pred_labels = (y_pred_prob > 0.5).astype(int)
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred_labels),
            "recall": recall_score(y_true, y_pred_labels, zero_division=0),
            "precision": precision_score(y_true, y_pred_labels, zero_division=0),
            "F1": f1_score(y_true, y_pred_labels, zero_division=0),
            "AUC": roc_auc_score(y_true, y_pred_prob),
        }
        return metrics

    def fit(self, epochs, seed=42):
        self.set_seed(seed)
        self.epochs = epochs

        for epoch in range(epochs):
            self.total_epochs += 1

            loss, all_preds, all_ys = self._mini_batch(validation=False)
            self.train_loss.append(loss)

            train_metrics = self._compute_metrics(all_ys, all_preds)

            self.train_acc.append(train_metrics["accuracy"])
            self.train_recall.append(train_metrics["recall"])
            self.train_precision.append(train_metrics["precision"])
            self.train_f1.append(train_metrics["F1"])
            self.train_auc.append(train_metrics["AUC"])

            with torch.no_grad():
                val_loss, val_preds, val_ys = self._mini_batch(validation=True)
                self.val_loss.append(val_loss)

                val_metrics = self._compute_metrics(val_ys, val_preds)

                self.val_acc.append(val_metrics["accuracy"])
                self.val_recall.append(val_metrics["recall"])
                self.val_precision.append(val_metrics["precision"])
                self.val_f1.append(val_metrics["F1"])
                self.val_auc.append(val_metrics["AUC"])

            if self.scheduler:
                self.scheduler.step(val_loss)

            if self.early_stopping_patience is not None:
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.early_stopping_counter = 0
                    self.best_model_state = self.model.state_dict()
                else:
                    self.early_stopping_counter += 1

                if self.early_stopping_counter >= self.early_stopping_patience:
                    print(f"Early stopping occurred at epoch {self.total_epochs}")
                    print(
                        f"Restoring model to best validation loss: {self.best_val_loss}"
                    )
                    self.model.load_state_dict(self.best_model_state)
                    break

        print("\n--- Training Completed ---")
        final_train_loss = self.train_loss[-1]
        final_val_loss = self.val_loss[-1]

        final_train_recall = self.train_recall[-1]
        final_val_recall = self.val_recall[-1]

        final_train_accuracy = self.train_acc[-1]
        final_val_accuracy = self.val_acc[-1]

        final_train_precision = self.train_precision[-1]
        final_val_precision = self.val_precision[-1]

        final_train_f1 = self.train_f1[-1]
        final_val_f1 = self.val_f1[-1]

        final_train_auc = self.train_auc[-1]
        final_val_auc = self.val_auc[-1]

        print(
            f"Final training loss = {final_train_loss:.4f}  |  Final validation loss = {final_val_loss:.4f}"
        )
        print(
            f"Final training accuracy = {final_train_accuracy:.4f}  |  Final validation accuracy = {final_val_accuracy:.4f}"
        )
        print(
            f"Final training recall = {final_train_recall:.4f}  |  Final validation recall = {final_val_recall:.4f}"
        )
        print(
            f"Final training precision = {final_train_precision:.4f}  |  Final validation precision = {final_val_precision:.4f}"
        )
        print(
            f"Final training F1 = {final_train_f1:.4f}  |  Final validation F1 = {final_val_f1:.4f}"
        )
        print(
            f"Final training AUC = {final_train_auc:.4f}  |  Final validation AUC = {final_val_auc:.4f}\n"
        )

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.as_tensor(X).float()
        with torch.no_grad():
            X_tensor = X_tensor.to(self.device)
            yhat = self.model(X_tensor)
            pred = torch.sigmoid(yhat)

        return pred.cpu()

    def save_model(self, filename):
        checkpoint = {
            "epoch": self.total_epochs,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": self.train_loss,
            "val_loss": self.val_loss,
        }

        torch.save(checkpoint, filename)

    def load_model(self, filename):
        checkpoint = torch.load(filename, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.total_epochs = checkpoint["epoch"]
        self.train_loss = checkpoint["loss"]
        self.val_loss = checkpoint["val_loss"]

        self.model.train()

    def plot_losses(self, save_path=None):
        fig = plt.figure(figsize=(10, 4))
        plt.plot(self.train_loss, label="Train Loss")
        plt.plot(self.val_loss, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")

        return fig

    def plot_recall(self, save_path=None):
        fig = plt.figure(figsize=(10, 4))
        plt.plot(self.train_recall, label="Train Recall")
        plt.plot(self.val_recall, label="Validation Recall")
        plt.xlabel("Epochs")
        plt.ylabel("Recall")
        plt.legend()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")

        return fig

    def plot_accuracy(self, save_path=None):
        fig = plt.figure(figsize=(10, 4))
        plt.plot(self.train_acc, label="Train Accuracy")
        plt.plot(self.val_acc, label="Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")

        return fig
