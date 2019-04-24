using CNTK;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;


//https://github.com/kvfrans/openai-cartpole/blob/master/cartpole-policygradient.py
public class VPGAgent
{
    const int LayerSize = 64;
    const float LearningRate = 0.005f;

    const string SavePolicyModelFileName = "policyModel";
    const string SaveValueModelFileName = "valueModel";
    const string SavePolicyTrainerFileName = "policyTrainer";
    const string SaveValueTrainerFileName = "valueTrainer";

    public VPGAgent(int stateSize, int actionSize, bool loadSavedModels)
    {
        for (int i = 0; i < actionSize; ++i)
        {
            m_normDist.Add(new EDA.Gaussian());
        }

        m_inputState = CNTKLib.InputVariable(new int[] { stateSize }, DataType.Float);

        CreatePolicyModel(stateSize, actionSize);
        CreateValueModel(stateSize);

        if (loadSavedModels)
        {
            if (File.Exists(SavePolicyModelFileName))
                m_policyModel.Restore(SavePolicyModelFileName);

            if (File.Exists(SaveValueModelFileName))
                m_valueModel.Restore(SaveValueModelFileName);

            if (File.Exists(SavePolicyTrainerFileName))
                m_policyTrainer.RestoreFromCheckpoint(SavePolicyTrainerFileName);

            if (File.Exists(SaveValueTrainerFileName))
                m_valueTrainer.RestoreFromCheckpoint(SaveValueTrainerFileName);
        }


    }

    private void CreatePolicyModel(int stateSize, int actionSize)
    {
        var l1 = CNTKLib.Tanh(Utils.Layer(m_inputState, LayerSize));
        var l2 = CNTKLib.Tanh(Utils.Layer(l1, LayerSize));

        var model = new VariableVector();
        var logProbList = new VariableVector();

        for (int i = 0; i < actionSize; ++i)
        {
            var mean = CNTKLib.Tanh(Utils.Layer(l2, 1));
            var std = CNTKLib.ReLU(Utils.Layer(l2, 1));
            m_mean.Add(mean);
            m_std.Add(std);

            model.Add(mean);
            model.Add(std);

            m_normalSamples.Add(CNTKLib.InputVariable(new int[] { 1 }, DataType.Float));

            logProbList.Add(Utils.GaussianProbabilityLayer(m_mean[i], m_std[i], m_normalSamples[i]));
        }

        var combinedLogProb = CNTKLib.Splice(logProbList, new Axis(0));

        m_policyAdvantage = CNTKLib.InputVariable(new int[] { 1 }, DataType.Float);
        m_policyLoss = CNTKLib.Negate(CNTKLib.ReduceSum(CNTKLib.ElementTimes(combinedLogProb, m_policyAdvantage), Axis.AllStaticAxes()));

        m_policyModel = CNTKLib.Splice(model, new Axis(0));

        var learningRate = new TrainingParameterScheduleDouble(LearningRate);
        var options = new AdditionalLearningOptions();
        options.gradientClippingThresholdPerSample = 1;
        options.l2RegularizationWeight = 0.01;

        var learner = new List<Learner>() { Learner.SGDLearner(m_policyModel.Parameters(), learningRate, options) };

        m_policyTrainer = Trainer.CreateTrainer(m_policyModel, m_policyLoss, m_policyLoss, learner);
    }

    private void CreateValueModel(int stateSize)
    {
        var l1 = CNTKLib.ReLU(Utils.Layer(m_inputState, LayerSize));
        var l2 = CNTKLib.ReLU(Utils.Layer(l1, LayerSize));

        m_valueModel = CNTKLib.ReLU(Utils.Layer(l2, 1));

        m_valueInput = CNTKLib.InputVariable(new int[] { 1 }, DataType.Float);

        m_valueLoss = CNTKLib.SquaredError(m_valueModel, m_valueInput);

        var learningRate = new TrainingParameterScheduleDouble(LearningRate);
        var options = new AdditionalLearningOptions();
        options.gradientClippingThresholdPerSample = 1;
        options.l2RegularizationWeight = 0.01;
        var learner = new List<Learner>() { Learner.SGDLearner(m_valueModel.Parameters(), learningRate, options) };

        m_valueTrainer = Trainer.CreateTrainer(m_valueModel, m_valueLoss, m_valueLoss, learner);
    }

    public float[] Act(float[] state, DeviceDescriptor device)
    {
        Value input = Value.CreateBatch<float>(new int[] { state.Length }, state, device);

        var inputDict = new Dictionary<Variable, Value>()
        {
            { m_inputState, input },
        };

        var outputDict = new Dictionary<Variable, Value>()
        {
            { m_policyModel.Output, null },
        };

        m_policyModel.Evaluate(inputDict, outputDict, device);

        var outputValue = outputDict[m_policyModel.Output];
        var denseData = outputValue.GetDenseData<float>(m_policyModel.Output)[0];

        float[] denseDataFloat = new float[denseData.Count];
        denseData.CopyTo(denseDataFloat, 0);

        m_normDist[0].Mean = denseDataFloat[0];
        m_normDist[0].StdDev = denseDataFloat[1];
        m_normDist[1].Mean = denseDataFloat[2];
        m_normDist[1].StdDev = denseDataFloat[3];

        var sample1 = Mathf.Clamp((float)m_normDist[0].Next(), -1.0f, 1.0f);
        var sample2 = Mathf.Clamp((float)m_normDist[1].Next(), -1.0f, 1.0f);

        return new float[2] { sample1, sample2 };
    }

    public float Predict(float[] state, DeviceDescriptor device)
    {
        Value input = Value.CreateBatch<float>(new int[] { state.Length }, state, device);

        var inputDict = new Dictionary<Variable, Value>()
        {
            { m_inputState, input },
        };

        var outputDict = new Dictionary<Variable, Value>()
        {
            { m_valueModel.Output, null },
        };

        m_valueModel.Evaluate(inputDict, outputDict, device);

        var outputValue = outputDict[m_valueModel.Output];
        var denseData = outputValue.GetDenseData<float>(m_valueModel.Output)[0];

        float[] denseDataFloat = new float[denseData.Count];
        denseData.CopyTo(denseDataFloat, 0);

        return UnityEngine.Mathf.Clamp(denseDataFloat[0], -1.0f, 1.0f);
    }

    public void TrainValueModel(float[] states, float[] futureRewards, DeviceDescriptor device)
    {
        Value input = Value.CreateBatch<float>(new int[] { 9 }, states, device);
        Value output = Value.CreateBatch<float>(new int[] { 1 }, futureRewards, device);

        var arguments = new Dictionary<Variable, Value>()
            {
                { m_inputState, input },
                { m_valueInput, output}
            };

        m_valueTrainer.TrainMinibatch(arguments, false, device);
        //Debug.Log("VLoss: " + m_valueTrainer.PreviousMinibatchLossAverage());
    }

    public void TrainPolicyModel(float[] states, float[] advantages, float[] actions1, float[] actions2, DeviceDescriptor device)
    {
        Value stateData = Value.CreateBatch<float>(new int[] { 9 }, states, device);
        Value sample1 = Value.CreateBatch<float>(new int[] { 1 }, actions1, device);
        Value sample2 = Value.CreateBatch<float>(new int[] { 1 }, actions2, device);
        Value output = Value.CreateBatch<float>(new int[] { 1 }, advantages, device);

        var arguments = new Dictionary<Variable, Value>()
            {
                { m_inputState, stateData },
                { m_normalSamples[0],  sample1},
                { m_normalSamples[1],  sample2},
                { m_policyAdvantage, output}
            };

        m_policyTrainer.TrainMinibatch(arguments, false, device);

        //Debug.Log("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!PLoss2: " + m_policyTrainer.PreviousMinibatchLossAverage());
    }

    public float GetPolicyModelLoss()
    {
        return (float)m_policyTrainer.PreviousMinibatchLossAverage();
    }

    public float GetValueModelLoss()
    {
        return (float)m_valueTrainer.PreviousMinibatchLossAverage();
    }

    public void SaveModel()
    {
        m_policyModel.Save(SavePolicyModelFileName);
        m_valueModel.Save(SaveValueModelFileName);
        m_policyTrainer.SaveCheckpoint(SavePolicyTrainerFileName);
        m_valueTrainer.SaveCheckpoint(SaveValueTrainerFileName);
    }

    Function m_policyModel;
    Function m_valueModel;

    Variable m_inputState;
    Variable m_policyAdvantage;

    private List<Function> m_mean = new List<Function>();
    private List<Function> m_std = new List<Function>();
    private List<Variable> m_normalSamples = new List<Variable>();
    private Function m_policyLoss;
    private Trainer m_policyTrainer;
    private Variable m_valueInput;
    private Function m_valueLoss;
    private Trainer m_valueTrainer;

    private List<EDA.Gaussian> m_normDist = new List<EDA.Gaussian>();
}
