class IterLoader:

    def __init__(self, dataloader):
        self._dataloader = dataloader
        # print('ooooo')
        # import time
        # t1 = time.time()
        self.iter_loader = iter(self._dataloader)
        # print(time.time() - t1)
        # import sys; sys.exit()


    def __next__(self):
        try:
            data = next(self.iter_loader)
        except StopIteration:
            self.iter_loader = iter(self._dataloader)
            data = next(self.iter_loader)

        return data

    def __len__(self):
        return len(self._dataloader)

    def __iter__(self):
        return self