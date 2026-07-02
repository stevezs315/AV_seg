import multiprocessing
import os
import time

import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from utils_pytorch import EarlyStopReduceLROnPlateau, save_model, save_to_csv
from factories import ModelFactory, LossesFactory
from train import clean_checkpoints


def learning_curves(training, validation, outfile):
    """Build learning curves for training and validation losses."""
    plt.rcParams["figure.figsize"] = [16, 9]
    fig, ax1 = plt.subplots(nrows=1, ncols=1, sharex=True)
    assert isinstance(ax1, Axes)

    x, y1 = zip(*training)
    ax1.plot(x, y1, 'b', label='training')

    x, y1 = zip(*validation)
    ax1.plot(x, y1, 'r', label='validation')

    ax1.set_yscale('log')
    ax1.legend()
    fig.savefig(outfile)
    plt.close(fig)


class R2Vessels:
    def __init__(
        self,
        base_channels=64,
        in_channels=3,
        out_channels=3,
        num_iterations=5,
        model='RRWNet',
        gpu_id=None,
        criterion='RRLoss',
        base_criterion='BCE3Loss',
        add_criterion=None,
        learning_rate=1e-4,
        checkpoint_path=None,
        scheduler_path=None,
        use_checkpoint=0,
    ):
        current = multiprocessing.current_process()
        self.process_id = str(current.pid)

        self.use_cuda = torch.cuda.is_available()
        if gpu_id is None:
            self.device = torch.device('cuda', 0)
            torch.cuda.set_device(0)
        else:
            self.device = torch.device('cuda', gpu_id)
            torch.cuda.set_device(gpu_id)

        self.checkpoint_path = checkpoint_path
        self.scheduler_path = scheduler_path
        self.use_checkpoint = use_checkpoint

        self.criterion_name = criterion
        self.base_criterion = base_criterion
        self.add_criterion = add_criterion
        losses_factory = LossesFactory()
        base_loss = losses_factory.create_class(base_criterion)
        add_loss = losses_factory.create_class(add_criterion) if add_criterion else None
        self.criterion = losses_factory.create_class(criterion, base_loss, add_loss)

        self.model_name = model
        self.model = ModelFactory().create_class(
            model,
            input_ch=in_channels,
            output_ch=out_channels,
            base_ch=base_channels,
            num_iterations=num_iterations,
        )
        if self.use_cuda:
            self.model.cuda()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
        )

        self.iter = 0
        self.test_count = 0

    def save_checkpoint(self, path):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'iter': self.iter,
            'test_count': self.test_count,
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.iter = checkpoint['iter']
        self.test_count = checkpoint['test_count']
        print(
            f"Checkpoint loaded from {self.checkpoint_path}\n"
            f"(Iteration: {self.iter}, test_count: {self.test_count})"
        )

    def _prepare_sample(self, sample):
        data = sample[1]
        retino = data[0].cuda(non_blocking=True)
        vessels = data[1].cuda(non_blocking=True)
        mask = data[2].cuda(non_blocking=True)
        return retino, vessels, mask

    def train_epoch(self, r2v_loader):
        total_loss = 0.0
        self.model.train()

        for sample in r2v_loader:
            retino, vessels, mask = self._prepare_sample(sample)
            retino.requires_grad_(True)
            vessels.requires_grad_(False)
            mask.requires_grad_(False)

            self.optimizer.zero_grad()
            imgs, predictions, cl_features = self.model(retino)
            loss = self.criterion(
                imgs, predictions, vessels, mask, cl_features, self.iter, superpixel_mask=None
            )
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            self.iter += 1

        avg_loss = total_loss / len(r2v_loader)
        pattern = '\n|{}| [PID: {}, {}, base: {}, add: {}] >> training epoch mean loss: {}'
        print(pattern.format(
            self.iter,
            self.process_id,
            self.model_name,
            self.base_criterion,
            self.add_criterion,
            avg_loss,
        ))

        return [avg_loss]

    def test(self, r2v_dataloader, prefix_to_save=None):
        with torch.no_grad():
            total_loss = 0.0
            self.model.eval()

            for sample in r2v_dataloader:
                try:
                    k = sample[0].numpy()[0]
                except AttributeError:
                    k = sample[0][0]

                retino, vessels, mask = self._prepare_sample(sample)
                imgs, predictions, cl_features = self.model(retino)
                loss = self.criterion(
                    imgs, predictions, vessels, mask, cl_features, self.iter, superpixel_mask=None
                )

                if prefix_to_save is not None:
                    self.criterion.save_predicted(predictions, prefix_to_save + str(k) + '.png')

                total_loss += loss.item()

            avg_loss = total_loss / len(r2v_dataloader)
            pattern = '|{}| [PID: {}, {}, {}] >> validation mean loss: {}'
            print(pattern.format(
                self.iter,
                self.process_id,
                self.model_name,
                self.criterion_name,
                avg_loss,
            ))

            return [avg_loss]

    def training(
        self,
        train_loader,
        test_loader,
        path_to_save,
        init_iter=0,
        save_period=25,
        scheduler_patience=25,
        stopping_patience=25,
    ):
        save_to_csv([['best_loss', 'iter']], os.path.join(path_to_save, 'best_loss.csv'))
        save_to_csv([['loss', 'iter']], os.path.join(path_to_save, 'train_loss.csv'))
        save_to_csv([['loss', 'iter']], os.path.join(path_to_save, 'test_loss.csv'))

        train_loss = []
        test_loss = []
        all_train_loss = []
        all_test_loss = []

        start_time = time.time()

        if self.use_checkpoint:
            print(f"Loading checkpoint from {self.checkpoint_path}...")
            self.load_checkpoint()
        else:
            self.iter = init_iter
            self.test_count = 0

        scheduler = EarlyStopReduceLROnPlateau(
            self.optimizer,
            self.model,
            path_to_save,
            factor=0.1,
            patience=scheduler_patience,
            patience_stopping=stopping_patience,
            verbose=True,
            cooldown=0,
            threshold=0,
            min_lr=1e-8,
            eps=0,
        )

        if self.use_checkpoint:
            print(f"Loading scheduler from {self.scheduler_path}...")
            scheduler._load_state(self.scheduler_path)

        while scheduler.training():
            save = self.test_count % save_period == 0

            if save:
                save_path = os.path.join(path_to_save, str(self.iter))
                os.makedirs(save_path, exist_ok=True)
                prefix_to_save = save_path + '/'
            else:
                prefix_to_save = None

            train_loss.append([self.iter] + self.train_epoch(train_loader))
            test_loss.append([self.iter] + self.test(test_loader, prefix_to_save))

            is_best = scheduler.step(test_loss[-1][1], self.iter)
            if is_best:
                save_to_csv(
                    [[str(test_loss[-1][1]), str(self.iter)]],
                    os.path.join(path_to_save, 'best_loss.csv'),
                )
                save_model(self.model, path_to_save + '/generator_best.pth')

            if save:
                self.save_checkpoint(os.path.join(path_to_save, f'checkpoint_{self.iter}.pth'))
                scheduler._save_state(os.path.join(path_to_save, f'scheduler_{self.iter}.pth'))
                clean_checkpoints(path_to_save)

                save_to_csv(train_loss, os.path.join(path_to_save, 'train_loss.csv'))
                save_to_csv(test_loss, os.path.join(path_to_save, 'test_loss.csv'))
                all_train_loss += train_loss
                all_test_loss += test_loss
                train_loss = []
                test_loss = []
                learning_curves(all_train_loss, all_test_loss, path_to_save + '/learning_curves.svg')

            self.test_count += 1

        if len(train_loss) > 0:
            save_to_csv(train_loss, os.path.join(path_to_save, 'train_loss.csv'))
        if len(test_loss) > 0:
            save_to_csv(test_loss, os.path.join(path_to_save, 'test_loss.csv'))

        save_model(self.model, path_to_save + '/generator_last.pth')
        self.save_checkpoint(os.path.join(path_to_save, 'checkpoint_final.pth'))
        if all_train_loss and all_test_loss:
            learning_curves(all_train_loss, all_test_loss, path_to_save + '/learning_curves.svg')

        total_time = time.time() - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        print(f"\nTraining completed. Total time: {hours}h {minutes}m {seconds}s")
