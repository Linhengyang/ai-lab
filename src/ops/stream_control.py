# customized design pattern for sync stream process

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, FIRST_COMPLETED
import typing as t

# 同步的流水线处理, 很多地方使用 列表生成式+生成器, 即
# [process_fn(item) for item in data_gen]
# 并发执行下是这样: 
# futures = [executor.submit(process_fn, item) for item in data_gen]
# for future in as_completed(futures):
#     result = future.result()
#     result_handler_fn(result)

# 但列表生成式没有背压设计, 即 for item in data_gen 会在主线程中, 无视下游的处理消费能力, 一直消费data_gen
# 若下游的处理消费能力跟不上, 那么就会使得 item 大量积压在内存中, 造成内存爆炸. 即 同步for-loop不作任何协调暂停
# 真正的背压设计只存在在 asyncio, RxPy, Streams 和 kafka-like queue. 这里提供一个可以pending for-loop消费的同步流式处理


def stream_parallel_process_with_pending(
    executor: ProcessPoolExecutor|ThreadPoolExecutor,
    data_gen: t.Generator,
    process_fn: t.Callable[[t.Any], t.Any],
    result_handler: t.Callable[[t.Any], None],
    max_pending: int = 8,
    process_args=(),
    result_handler_args=(),
    ):
    '''
    流式并发处理生成器中的任务, 控制内存占用

    Args:
        executor: 进程池 or 线程池
        data_gen: generator, 在主线程中 逐批生成任务数据 item
        process_fn: 对每一批的处理任务(进程池/线程池中执行)
        result_handler: 对任务结果的处理(在主线程中执行), 要求return None, 即一个副作用函数(修改某个容器而不是返回值)
        max_pending: 最大挂起任务数(根据内存控制)
        process_args: 可被 result = process_fn(item, *process_args) 的方式调用
        result_handler_args: 可被 result_handler(result, *result_handler_args)  的方式调用. 其中应该包括一个容器, 以记录副作用
    
    当 executor 为 线程池 时, item 由 data_gen 在主线程生成, 可共享给工作线程, 所以 item 没有序列化要求. process_args 也一样, 会由
    主线程共享给工作线程, 所以 process_args 也不需要序列化.
    process_fn 也不需要序列化, 同时由于可以从 主线程 共享, 所以它还可以是 lambda, 嵌套函数, 闭包
    但它必须符合以下:
        1. 线程安全. 若 process_fn 涉及 写共享资源, 那么必须要加线程锁; 更推荐的做法是 process_fn 只读参数、只返回结果、无外部副作用
        2. 若希望得到加速效果, 那么 process_fn 应该有效绕开 GIL, 否则 GIL 会有效限制 py解释器的多线程并发
    
    当 executor 为 进程池 时, item 由 data_gen 在 父进程主线程 生成, 必须要 序列化pickle给子进程. process_args 也一样, 需要从父进程
    序列化给子进程. 同时 process_fn 返回的结果 result 也需要 pickle 序列化到 父进程, 再由 result_handler 收集. 总而言之,
    process_fn 需要满足:
        1. 自身可序列化, 即必须是顶层函数, 不可以是 lambda、嵌套函数
        2. process_fn 所需 数据 item(data_gen) 和 参数 process_args, 以及 返回的结果 result 都必须 可序列化
    '''
    futures = set()

    for item in data_gen:
        # 提交任务
        future = executor.submit(process_fn, item, *process_args)
        futures.add(future)

        # 当任务队列长度达到 max_pending 时, 暂停提交任务, 等到部分已提交的任务完成并处理
        if len(futures) >= max_pending:
            # wait会返回已经处理好的 futures(作为done) 和 尚未处理好的 futures(作为futures)
            # wait 阻塞主线程, 等待至少一个future完成
            done, futures = wait(futures, return_when=FIRST_COMPLETED) # 直接更新 futures, 从而已经完成的 future 不再在内
            for f in done:
                result = f.result()
                if result_handler:
                    result_handler(result, *result_handler_args)

    # 收尾, 处理剩下的任务
    for f in futures:
        result = f.result()
        if result_handler:
            result_handler(result, *result_handler_args)