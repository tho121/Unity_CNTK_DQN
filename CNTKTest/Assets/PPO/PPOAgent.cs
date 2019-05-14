using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CNTK;
using UnityEngine;

class PPOAgent
{
    const int LayerSize = 32;

    const float LearningRate = 0.003f;
    const float AdamBeta1 = 0.9f;
    const float AdamBeta2 = 0.999f;
    const float Epsilon = 0.0000001f;
    float EntropyCoefficient = 0.01f;

    const string SaveModelFileName = "ppo_model.model";
    const string SaveTrainerFileName = "ppo_trainer.train";

    public PPOAgent(int stateSize, int actionSize, int maxSteps, DeviceDescriptor device, bool loadSavedModels = false)
    {
        m_stateSize = stateSize;
        m_actionSize = actionSize;

        for (int i = 0; i < actionSize; ++i)
        {
            m_normDist.Add(new EDA.Gaussian());
        }

        //state, actions, probabilities, target value, advantage
        m_memory = new Memory(stateSize + actionSize + actionSize + 2, maxSteps);

        m_inputState = CNTKLib.InputVariable(new int[] { stateSize }, DataType.Float, "State");
        m_targetValue = CNTKLib.InputVariable(new int[] { 1 }, DataType.Float, "TargetValue");
        m_inputAction = CNTKLib.InputVariable(new int[] { actionSize }, DataType.Float, "Actions");
        m_inputOldProb = CNTKLib.InputVariable(new int[] { actionSize }, DataType.Float, "Probs");
        m_advantage = CNTKLib.InputVariable(new int[] { 1 }, DataType.Float, "Advantage");
        m_entropyCoefficient = CNTKLib.InputVariable(new int[] { 1 }, DataType.Float, "EntropyCoeff");

        CreateModel(actionSize);

        if (loadSavedModels)
        {
            if (File.Exists(SaveModelFileName))
                m_modelLoss.Restore(SaveModelFileName);

            if (File.Exists(SaveTrainerFileName))
                m_trainer.RestoreFromCheckpoint(SaveTrainerFileName);
        }
    }

    private void CreateModel(int actionSize)
    {
        var l1 = CNTKLib.ReLU(Utils.Layer(m_inputState, LayerSize));
        //var l2 = CNTKLib.ReLU(Utils.Layer(l1, LayerSize));
        //var l3 = CNTKLib.ReLU(Utils.Layer(l2, LayerSize));
        //var l2b = CNTKLib.ReLU(Utils.Layer(l2, LayerSize));

        m_means = CNTKLib.Tanh(Utils.Layer(l1, actionSize));

        //https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#stochastic-policies
        //Note that in both cases we output log standard deviations instead of standard deviations directly. 
        //This is because log stds are free to take on any values in (-\infty, \infty), while stds must be nonnegative.
        //It’s easier to train parameters if you don’t have to enforce those kinds of constraints. 
        //The standard deviations can be obtained immediately from the log standard deviations by exponentiating them, so we do not lose anything by representing them this way.

        m_logstds = new Parameter(new int[] { actionSize }, DataType.Float, CNTKLib.ConstantInitializer(-0.5));
        //m_stds = CNTKLib.ReLU(Utils.Layer(l2b, actionSize));

        m_policyModel = CNTK.Function.Combine(new Variable[] { m_means, m_logstds });

        var vl1 = CNTKLib.ReLU(Utils.Layer(m_inputState, LayerSize));
        //var vl2 = CNTKLib.ReLU(Utils.Layer(vl1, LayerSize));
        m_valueModel = Utils.Layer(vl1, 1);
   
        var valueLoss = CNTKLib.ElementTimes(CNTKLib.SquaredError(m_targetValue, m_valueModel.Output), Constant.Scalar(DataType.Float, 1.0f));

        //entropy loss
        //0.5 + 0.5 * math.log(2 * math.pi) + torch.log(self.scale)
        //var entropy = CNTKLib.Plus(Constant.Scalar(DataType.Float, 0.5f), Constant.Scalar(DataType.Float, Mathf.Log(2.0f * UnityEngine.Mathf.PI) * 0.5f));
        //entropy = CNTKLib.Plus(m_logstds, entropy);
        //entropy = CNTKLib.ReduceMean(entropy, Axis.AllStaticAxes());

        //var std = CNTKLib.Exp(m_logstds);

        ////https://math.stackexchange.com/questions/1804805/how-is-the-entropy-of-the-normal-distribution-derived
        //var entropy = CNTKLib.Log(CNTKLib.ElementTimes(CNTKLib.ElementTimes(std, std), Constant.Scalar(DataType.Float, 2.0f * Mathf.PI * 2.7182818285)));
        //entropy = CNTKLib.ElementTimes(entropy, Constant.Scalar(DataType.Float, 0.5));
        //entropy = CNTKLib.ReduceSum(entropy, Axis.AllStaticAxes());

        var entropy = CNTKLib.ElementTimes(Constant.Scalar(DataType.Float, 2 * Mathf.PI * 2.7182818285), CNTKLib.Pow(CNTKLib.Exp(m_logstds), Constant.Scalar(DataType.Float, 2.0f)));
        entropy = CNTKLib.ElementTimes(Constant.Scalar(DataType.Float, 0.5), entropy);
        entropy = CNTKLib.ReduceSum(entropy, Axis.AllStaticAxes());

        //probability
        var actionProb = Utils.NormalLogProbabilityLayer( m_means, m_logstds, m_inputAction);

        var probRatio = CNTKLib.ElementDivide(actionProb, m_inputOldProb + Constant.Scalar(DataType.Float, 0.000001f));  //CNTKLib.Exp(CNTKLib.Minus(actionProb, m_inputOldProb + Constant.Scalar(DataType.Float, 0.0000000001f)));

        var constant1 = Constant.Scalar(DataType.Float, 1.0f);
        var constantClipEpsilon = Constant.Scalar(DataType.Float, 0.2f);

        var clip = CNTKLib.Clip(probRatio, CNTKLib.Minus(constant1, constantClipEpsilon), CNTKLib.Plus(constant1, constantClipEpsilon));

        var p_opt_a = CNTKLib.ElementTimes(probRatio, m_advantage);
        var p_opt_b = CNTKLib.ElementTimes(clip, m_advantage);

        var policyLoss = CNTKLib.ReduceMean(CNTKLib.ElementMin(p_opt_a, p_opt_b, "min"), Axis.AllStaticAxes());

        var finalLoss = CNTKLib.Minus(policyLoss, valueLoss);
        m_modelLoss = CNTKLib.Negate(CNTKLib.Plus(finalLoss, CNTKLib.ElementTimes(m_entropyCoefficient, entropy)));

        var options = new AdditionalLearningOptions();
        //options.gradientClippingThresholdPerSample = 0.5f;

        var learner = CNTKLib.AdamLearner(new ParameterVector(finalLoss.Parameters().ToArray()),
            new TrainingParameterScheduleDouble(LearningRate),
            new TrainingParameterScheduleDouble(AdamBeta1),
            true,
            new TrainingParameterScheduleDouble(AdamBeta2),
            Epsilon);
            //false,
            //options);


        //var learningRate = new TrainingParameterScheduleDouble(LearningRate);
        
        //options.l2RegularizationWeight = 0.01;

        //var sgdlearner = Learner.SGDLearner(m_modelLoss.Parameters(), learningRate, options);

        m_trainer = CNTK.Trainer.CreateTrainer(m_modelLoss, m_modelLoss, null, new List<Learner>() { learner });
    }

    public float[] Act(float[] state, DeviceDescriptor device, out float[] probabilities)
    {
        Value input = Value.CreateBatch<float>(new int[] { state.Length }, state, device);

        var inputDict = new Dictionary<Variable, Value>()
        {
            { m_inputState, input },
        };

        var outputDict = new Dictionary<Variable, Value>()
        {
            { m_means.Output, null },
            { m_logstds.Output, null },
        };

        m_policyModel.Evaluate(inputDict, outputDict, device);

        var meansValue = outputDict[m_means.Output];
        var meansDenseData = meansValue.GetDenseData<float>(m_means.Output)[0];

        var logstdsValue = outputDict[m_logstds.Output];
        var logstdsDenseData = logstdsValue.GetDenseData<float>(m_logstds.Output)[0];

        float[] denseDataFloat = new float[m_actionSize];
        meansDenseData.CopyTo(denseDataFloat, 0);

        m_normDist[0].Mean = denseDataFloat[0];
        m_normDist[1].Mean = denseDataFloat[1];

        logstdsDenseData.CopyTo(denseDataFloat, 0);

        m_normDist[0].StdDev = Mathf.Max(Mathf.Exp(denseDataFloat[0]), Epsilon);
        m_normDist[1].StdDev = Mathf.Max(Mathf.Exp(denseDataFloat[1]), Epsilon);

        var sample1 =(float)m_normDist[0].Next();
        var sample2 = (float)m_normDist[1].Next();

        //probabilities = new float[2] { (float)m_normDist[0].LogProbabilityFunction(sample1), (float)m_normDist[1].LogProbabilityFunction(sample2)};
        probabilities = new float[2] { Utils.NormalLogProbability((float)m_normDist[0].Mean, denseDataFloat[0], sample1),
                                        Utils.NormalLogProbability((float)m_normDist[1].Mean, denseDataFloat[1], sample2) };

        //sample1 = Mathf.Clamp(sample1, -1.0f, 1.0f);
        //sample2 = Mathf.Clamp(sample2, -1.0f, 1.0f);

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

        return denseDataFloat[0];
        //return UnityEngine.Mathf.Clamp(denseDataFloat[0], -1.0f, 1.0f);
    }

    public void AddExperience(float[] state, float[] actions, float[] probabilities, float targetValue, float advantage)
    {
        var experience = new float[state.Length + actions.Length + probabilities.Length + 2];   //2 -> targetValue and advantage

        Array.Copy(state, experience, state.Length);

        int rollingIndex = state.Length;

        Array.Copy(actions, 0, experience, rollingIndex, actions.Length);
        rollingIndex += actions.Length;

        Array.Copy(probabilities, 0, experience, rollingIndex, probabilities.Length);
        rollingIndex += probabilities.Length;

        experience[rollingIndex] = targetValue;
        rollingIndex += 1;

        experience[rollingIndex] = advantage;

        m_memory.Add(experience);
    }

    public void Train(DeviceDescriptor device)
    {
        float[] samples = m_memory.GetAllSamples();

        int experienceSize = m_memory.GetExperienceSize();

        var sampleSize = samples.Length / experienceSize;

        List<float> states = new List<float>(m_stateSize * sampleSize);
        List<float> actions = new List<float>(m_actionSize * sampleSize);
        List<float> probabilities = new List<float>(m_actionSize * sampleSize);
        List<float> targetValues = new List<float>(sampleSize);
        List<float> advantages = new List<float>(sampleSize);
        List<float> entropyCoeff = new List<float>(sampleSize);

        //state, actions, probabilities, target value, advantage

        for (int i = 0; i < sampleSize; ++i)
        {
            int start = i * experienceSize;
            for (int j = 0; j < m_stateSize; ++j)
            {
                states.Add(samples[start + j]);
            }

            for (int j = 0; j < m_actionSize; ++j)
            {
                actions.Add(samples[start + m_stateSize + j]);
            }

            for (int j = 0; j < m_actionSize; ++j)
            {
                probabilities.Add(samples[start + m_stateSize + m_actionSize + j]);
            }

            targetValues.Add(samples[start + m_stateSize + m_actionSize + m_actionSize]);
            advantages.Add(samples[start + m_stateSize + m_actionSize + m_actionSize + 1]);
            entropyCoeff.Add(EntropyCoefficient);
        }

        Value stateData = Value.CreateBatch<float>(new int[] { m_stateSize }, states.ToArray(), device);
        Value actionsData = Value.CreateBatch<float>(new int[] { m_actionSize }, actions.ToArray(), device);
        Value probabilitiesData = Value.CreateBatch<float>(new int[] { m_actionSize }, probabilities.ToArray(), device);
        Value targetValuesData = Value.CreateBatch<float>(new int[] { 1 }, targetValues.ToArray(), device);
        Value advantagesData = Value.CreateBatch<float>(new int[] { 1 }, advantages.ToArray(), device);
        Value entropyCoefficient = Value.CreateBatch<float>(new int[] { 1 }, entropyCoeff.ToArray(), device);

        var arguments = new Dictionary<Variable, Value>()
            {
                { m_inputState, stateData },
                { m_inputAction,  actionsData},
                { m_inputOldProb,  probabilitiesData},
                { m_targetValue, targetValuesData},
                { m_advantage, advantagesData},
                { m_entropyCoefficient,  entropyCoefficient},
            };

        m_trainer.TrainMinibatch(arguments, false, device);

        //clear episode memory
        //m_memory.ClearMemory();
        //EntropyCoefficient *= 0.9999f;
    }

    public float GetModelLoss()
    {
        return (float)m_trainer.PreviousMinibatchLossAverage();
    }

    public void SaveModel()
    {
        m_modelLoss.Save(SaveModelFileName);
        m_trainer.SaveCheckpoint(SaveTrainerFileName);
    }

    public Memory GetMemory()
    {
        return m_memory;
    }

    private Variable m_inputState;
    private Variable m_targetValue;
    private Variable m_inputAction;
    private Variable m_inputOldProb;
    private Variable m_advantage;
    private Variable m_entropyCoefficient;
    private Function m_policyModel;
    private Function m_valueModel;
    private Function m_modelLoss;

    private Function m_means;
    private Function m_logstds;

    private List<EDA.Gaussian> m_normDist = new List<EDA.Gaussian>();
    private Trainer m_trainer;

    private Memory m_memory;
    private int m_stateSize;
    private int m_actionSize;
    
}

