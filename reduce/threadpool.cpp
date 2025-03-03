#include<thread>
#include<condition_variable>
#include<mutex>
#include<queue>
#include<functional>
#include<atomic>
#include<future>
#include<vector>
#include<numeric>
#include<algorithm>
#include<iostream>
#include<type_traits>


class ThreadPool {



public:
    ThreadPool(int num):num_threads(num),stop(false) {
        for(int i = 0;i< num_threads; ++i) {
            threads.emplace_back([this]()
            {
                while(true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->mutex);
                        this->cv.wait(lock, [this](){return this->stop || !this->tasks.empty();});
                        if(this->stop && this->tasks.empty()) break;
                        task = std::move(this->tasks.front());
                        tasks.pop();
                    }
                    task();
                }
            }
            );
        }
    }
    template<typename F, typename... Args>
    auto addTask(F && func, Args&&... args)
    {
        using result_type = std::invoke_result_t<F, Args...>;
        auto task = std::make_shared<std::packaged_task<result_type()>>(
            std::bind(std::forward<F>(func), std::forward<Args>(args)...)
        );
        auto result = task->get_future();
        {
            std::unique_lock<std::mutex> lock(mutex);
            tasks.emplace([=](){(*task)();});
        }
        cv.notify_one();
        return result;
    }

    ~ThreadPool() {
        stop = true;
        cv.notify_all();
        for(auto & thread: threads) {
            thread.join();
        }
    }

private:
    int num_threads;
    std::vector<std::thread> threads;
    std::queue<std::function<void()>> tasks;
    std::mutex mutex;
    std::condition_variable cv;
    std::atomic<bool> stop;
};


template<typename Iterator,typename T>
T  parallel_accumulate(Iterator begin,Iterator end,T init)
{
    unsigned long const length = std::distance(begin,end);
    if(!length) return init;
    unsigned long const blockSize = 200000;
    unsigned int  const numBlocks = (length + blockSize - 1) / blockSize;
    std::vector<std::future<T>> futures(numBlocks - 1);
    std::cout<<"hardware_concurrency: "<<std::thread::hardware_concurrency()<<std::endl;
    int numThreads = std::thread::hardware_concurrency();
    ThreadPool pool(numThreads);
    Iterator blockStart = begin;
    for(int i = 0; i < (numBlocks - 1); ++i) {
        Iterator blockEnd = blockStart;
        std::advance(blockEnd,blockSize);
        futures[i] = pool.addTask(std::accumulate<Iterator,T>,blockStart,blockEnd,init);
        blockStart = blockEnd;
    }
    T lastResult = std::accumulate(blockStart,end,init);
    T result = init;
    for(int i = 0; i < (numBlocks - 1); ++i) {
        result += futures[i].get();
    }
    result += lastResult;
    return result;
}

void test()
{
    std::vector<int> v(500000000,1);
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    int sum = std::accumulate(v.begin(),v.end(),0);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "std::accumulate: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    std::cout << sum << std::endl;
    start = std::chrono::steady_clock::now();
    sum = parallel_accumulate(v.begin(),v.end(),0);
    end = std::chrono::steady_clock::now();
    std::cout << "parallel_accumulate: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    std::cout << sum << std::endl;
}

int main()
{
    test();
    return 0;
}

