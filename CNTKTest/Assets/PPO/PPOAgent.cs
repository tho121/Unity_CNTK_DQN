using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CNTK;

class PPOAgent
{
    const int LayerSize = 64;

    public PPOAgent(int stateSize, int actionSize, DeviceDescriptor device)
    {
        for (int i = 0; i < actionSize; ++i)
        {
            m_normDist.Add(new EDA.Gaussian());
        }

        m_inputState = CNTKLib.InputVariable(new int[] { stateSize }, DataType.Float);
        m_targetValue = CNTKLib.InputVariable(new int[] { 1 }, DataType.Float);
        m_inputAction = CNTKLib.InputVariable(new int[] { actionSize }, DataType.Float);
        m_inputOldProb = CNTKLib.InputVariable(new int[] { actionSize }, DataType.Float);
        m_advantage = CNTKLib.InputVariable(new int[] { 1 }, DataType.Float);
        m_entropyCoeffient = CNTKLib.InputVariable(new int[] { 1 }, DataType.Float);

        CreatePolicyModel(stateSize, actionSize);
        CreateValueModel(stateSize);
    }

    private void CreatePolicyModel(int stateSize, int actionSize)
    {
        var l1 = CNTKLib.Tanh(Utils.Layer(m_inputState, LayerSize));
        var l2 = CNTKLib.Tanh(Utils.Layer(l1, LayerSize));

        m_means = Utils.Layer(l2, actionSize);
        m_stds = CNTKLib.Exp( new Parameter(new int[] { actionSize }, DataType.Float, CNTKLib.ConstantInitializer(0)));

        m_policyModel = CNTK.Function.Combine(new Variable[] { m_means, m_stds });

        var vl1 = CNTKLib.ReLU(Utils.Layer(m_inputState, LayerSize));
        var vl2 = CNTKLib.ReLU(Utils.Layer(vl1, LayerSize));
        m_valueModel = CNTKLib.Tanh(Utils.Layer(vl2, LayerSize));

        var valueLoss = CNTKLib.SquaredError(m_valueModel.Output, m_targetValue);

        //entropy loss
        var temp = CNTKLib.ElementTimes(Constant.Scalar(DataType.Float, 2 * UnityEngine.Mathf.PI * 2.7182818285), m_stds);
        temp = CNTKLib.ElementTimes(Constant.Scalar(DataType.Float, 0.5), temp);
        var entropy = CNTKLib.ReduceSum(temp, Axis.AllStaticAxes());
        //probability
        var actionProb = Utils.GaussianLogProbabilityLayer(m_means, m_stds, m_inputAction);

        var probRatio = CNTKLib.ElementDivide(actionProb, m_inputOldProb + Constant.Scalar(DataType.Float, 0.0000000001f));

        var constant1 = Constant.Scalar(DataType.Float, 1.0f);
        var constantClipEpsilon = Constant.Scalar(DataType.Float, 0.1f);
        var p_opt_a = CNTKLib.ElementTimes(probRatio, m_advantage);
        var p_opt_b = CNTKLib.ElementTimes(
                CNTKLib.Clip(probRatio, 
                CNTKLib.Minus(constant1, constantClipEpsilon), constant1 + constantClipEpsilon), 
                m_advantage);

        var policyLoss = CNTKLib.Minus(constant1, CNTKLib.ReduceMean(CNTKLib.ElementMin(p_opt_a, p_opt_b, "min"), Axis.AllStaticAxes()));

        var finalLoss = CNTKLib.Plus(policyLoss, valueLoss);
        finalLoss = CNTKLib.Minus(finalLoss, CNTKLib.ElementTimes(m_entropyCoeffient, entropy));
    }

    private void CreateValueModel(int stateSize)
    {
        throw new NotImplementedException();
    }

    

    Variable m_inputState;
    Variable m_targetValue;
    Variable m_inputAction;
    Variable m_inputOldProb;
    private Variable m_advantage;
    private Variable m_entropyCoeffient;
    Function m_policyModel;
    Function m_valueModel;

    Function m_means;
    Function m_stds;

    private List<EDA.Gaussian> m_normDist = new List<EDA.Gaussian>();
}

