"""
相关代码注释请阅读相应的类设计文档
实现参考了 https://github.com/dieuwkehupkes/Tempotron

"""
import numpy as np
from matplotlib import pyplot as plt


class Tempotron:

    def __init__(self, w, v_rest, tau, tau_s, thred_hold, max_iter=1, alpha=0.1,mini_batch=50):
        #part1:基本参数
        self.w = w;
        self.v_rest = v_rest;
        self.tau = tau;
        self.tau_s = tau_s;
        self.thred_hold = thred_hold;
        self.v_0 = self.ComputeV_0();
        #论文method部分所提到的参数设置，但是效果并不是特别好
        #self.lamb = 0.0001 / self.v_0;
        self.lamb=1;

        #part2:batch训练参数
        self.max_iter = max_iter;
        self.alpha = alpha;
        self.mini_batch=mini_batch;


    def K(self, t, ti, v_0=1):
        if (t < ti):
            return 0;
        return v_0 * (np.exp(-(t - ti) / self.tau) - np.exp(-(t - ti) / self.tau_s));

    def ComputeV_0(self):
        t_max = (self.tau * self.tau_s * np.log(self.tau / self.tau_s)) / (self.tau - self.tau_s);
        v_max = self.K(t_max, 0);
        return 1.0 / v_max;

    def ComputeContributions(self, spike_input, t):
        sz = len(spike_input);
        ans = np.zeros(sz);
        for i in range(sz):
            for spike_time in spike_input[i]:
                ans[i] += self.K(t, spike_time, self.v_0);
        return ans;

    def ComputeMembranePotential(self, spike_input, t):
        res=self.ComputeContributions(spike_input, t);
        #print(res)
        res=res* self.w;
        res=res.sum()+self.v_rest;
        return res

    def GetOutput(self, t_start, t_end, spike_input, interval=0.1):
        t = list();
        v = list();
        for i in np.arange(t_start, t_end, interval):
            t.append(i);
            #print(type(self.ComputeMembranePotential(spike_input, i)));
            v.append(self.ComputeMembranePotential(spike_input, i));
        return (t, v);


    def PlotVT(self, t_start, t_end, spike_input, interval=0.1):
        (t, v) = self.GetOutput(t_start, t_end, spike_input, interval);
        plt.xlabel('Time (ms)');
        plt.ylabel('V(t)');
        ymax = max(max(v), self.thred_hold) + 0.1;
        ymin = min(min(v), -self.thred_hold) - 0.1;
        plt.axis([-10,500,ymin,ymax]);
        plt.axhline(y=self.thred_hold, linestyle='--', color='k');
        plot = plt.plot(t, v);
        plt.show()

    # ComputeTmax应该有两种很多求解的方式
    # 1.根据参考资料导数等于0来求解析解解
    # 2.离散求近似解


    def ComputeTmax1(self, input_spike):
        log_tts = np.log(self.tau / self.tau_s)
        tmp = [(time, id) for id in range(len(input_spike)) for time in input_spike[id]]
        tmp.sort()
        spikes = [(one[0], self.w[one[1]]) for one in tmp]
        times = np.array([spike[0] for spike in spikes])
        weights = np.array([spike[1] for spike in spikes])
        sum_tau = (weights * np.exp(times / self.tau)).cumsum()
        sum_tau_s = (weights * np.exp(times / self.tau_s)).cumsum()
        div = sum_tau_s / sum_tau
        boundary_cases = div < 0
        div[boundary_cases] = 10
        tmax_list = self.tau * self.tau_s * (log_tts + np.log(div)) / (self.tau - self.tau_s)
        tmax_list[boundary_cases] = times[boundary_cases]
        vmax_list = np.array([self.ComputeMembranePotential(input_spike,t) for t in tmax_list])
        tmax = tmax_list[vmax_list.argmax()]
        return tmax
    def ComputeTmax2(self,input_spike):
        return ;

    # output表示是否发放脉冲
    def AdaptWeight(self, input, label):
        t_max = self.ComputeTmax1(input);
        v_max = self.ComputeMembranePotential(input, t_max);
        if ((v_max > self.thred_hold) == label):
            return False;
        dw = self.ComputeContributions(input, t_max) * self.alpha * self.lamb;
        if (label == 1):
            self.w += dw;
        else:
            self.w -= dw;
        return True;

    def Train(self, train_data):
        for i in range(self.max_iter):
            for j in range(self.mini_batch):
                rand_idx=np.random.permutation()
                cnt = 0;
                for i in range(len(train_data)):
                    (data, label) = train_data[i];
                    t_max = self.ComputeTmax1(data);
                    v_max = self.ComputeMembranePotential(data, t_max);
                    if ((v_max >= self.thred_hold) == label):
                        cnt += 1
                print('%d/%d epoches,accuracy is %f%%' % (i, self.max_iter,i,100*cnt / len(train_data)))
                for i in range(len(train_data)):
                    (data, label)=train_data[i]
                    self.AdaptWeight(data, label)
                        #print("adapt weight for image %d" %i);

