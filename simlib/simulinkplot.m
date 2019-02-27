function varargout = simulinkplot( varargin )
    %����Simulink�ķ���������ͼ��
    %   �����������Ϊ����������ʱ�����У���һ���ź�������һ���źţ�
    %   �ڶ����ź������ڶ����źţ�����
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



