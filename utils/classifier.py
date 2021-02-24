import torch
from torch import nn
from tqdm import tqdm


class Classifier(nn.Module):
    def __init__(self, model=None):
        super(Classifier, self).__init__()
        if model is not None:
            self.model = model
        else:
            self.model = None

    def forward(self, x):
        if self.model is None:
            raise NotImplementedError
        else:
            return self.model.forward(x)

    def predict(self, x):
        return [self.forward(x)]

    def fit(self, loader, loss_fcn, optimizer, nb_epochs=10, **kwargs):

        if isinstance(loss_fcn, (torch.nn.CrossEntropyLoss, torch.nn.NLLLoss, torch.nn.MultiMarginLoss)):
            reduce_labels = True
        else:
            assert 0
        # Start training
        for i in range(nb_epochs):
            pbar = tqdm(loader)
            for i_batch, o_batch in pbar:
                i_batch, o_batch = i_batch.to('cuda'), o_batch.to('cuda')
                optimizer.zero_grad()
                # Perform prediction
                model_outputs = self.forward(i_batch)
                # Form the loss function
                loss = loss_fcn(model_outputs, o_batch)
                loss.backward()
                optimizer.step()
                pbar.set_description("epoch {:d}".format(i))

    def advfit(self, loader, loss_fcn, optimizer, attack, epsilon, nb_epochs=10, ratio=0.5, **kwargs):
        import foolbox as fb

        assert (0 <= ratio <= 1), "ratio must be between 0 and 1"
        if isinstance(loss_fcn, (torch.nn.CrossEntropyLoss, torch.nn.NLLLoss, torch.nn.MultiMarginLoss)):
            reduce_labels = True
        else:
            assert 0

        # Start training
        for _ in range(nb_epochs):
            pbar = tqdm(loader)
            # Shuffle the examples
            for i_batch, o_batch in pbar:
                i_batch, o_batch = i_batch.to('cuda'), o_batch.to('cuda')

                self.eval()
                fmodel = fb.PyTorchModel(self, bounds=(0, 1))
                adv_batch, _ = attack(fmodel, i_batch, o_batch, epsilon=epsilon, **kwargs)
                self.train()

                optimizer.zero_grad()
                # Perform prediction
                model_outputs = self.forward(i_batch)
                adv_outputs = self.forward(adv_batch)
                loss = (1 - ratio) * loss_fcn(model_outputs, o_batch) + ratio * loss_fcn(adv_outputs, o_batch)

                # Actual training
                loss.backward()
                optimizer.step()
                # pbar.set_description()

    def save(self, filename, path):
        """
        Save a model to file in the format specific to the backend framework.

        :param filename: Name of the file where to store the model.
        :type filename: `str`
        :param path: Path of the folder where to store the model. If no path is specified, the model will be stored in
                     the default data location of the library `ART_DATA_PATH`.
        :type path: `str`
        :return: None
        """
        import os
        assert(path is not None)

        full_path = os.path.join(path, filename)
        folder = os.path.split(full_path)[0]
        if not os.path.exists(folder):
            os.makedirs(folder)
        torch.save(self.state_dict(), full_path + ".model")

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = False