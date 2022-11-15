from process_group import Worker, DistributedRunner


class SimpleWorker(Worker):
    def __init__(self):
        pass

    def __call__(self, id, n_processes):
        print(f'Worker {id + 1}/{n_processes} started!')


if __name__ == '__main__':
    def spawn_worker(id):
        return SimpleWorker()

    runner = DistributedRunner(
        n_processes=2,
        spawn_worker=spawn_worker,
        master_addr='127.0.0.1',
        master_port='29500',
    )

    runner()
