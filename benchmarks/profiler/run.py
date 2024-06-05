from profiler import pre_hook,post_hook

def run():
    pre_hook()
    from torch.distributed.run import main
    main()
    post_hook()

if __name__ == '__main__':
    run()


