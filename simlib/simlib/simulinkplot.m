function varargout = simulinkplot( varargin )
    %生成Simulink的仿真结果并绘图。
    %   输入参数依次为：仿真名，时间序列，第一个信号名，第一个信号，
    %   第二个信号名，第二个信号，……
    ds = Simulink.SimulationData.Dataset();
    ds.Name = varargin{1};
    t = varargin{2};
    for i=3:2:nargin
        name = varargin{i};
        data = varargin{i+1};
        ts = timeseries(data,t);
        ts.Name = name;
        sig = Simulink.SimulationData.Signal();
        sig.Name = name;
        sig.Values = ts;
        ds = ds.addElement(sig);    
    end
    eval(char(string(ds.Name) + '=ds;'));
    eval(char('simplot('+string(ds.Name)+');'));
    if nargout == 1
        varargout{1} = ds;
    end
end



