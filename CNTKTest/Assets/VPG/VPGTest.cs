using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using CNTK;
using System;
using System.IO;

public class VPGTest : MonoBehaviour {

    public BallEnvironment env;
    
    public const int MaxEpisodes = 50000;
    public const int MaxTimeSteps = 10000;  //at fixed timestep 0.02 -> 200 seconds
    public const float Gamma = 0.9555f;
    
    public GraphUI graph;

    public bool LoadSavedModels = true;
    public bool SaveTraining = true;

    // Use this for initialization
    void Start () {

        env.selfReseting = false;

        m_device = DeviceDescriptor.UseDefaultDevice();
        print($"Hello from CNTK for {m_device.Type} only!");

        m_agent = new VPGAgent(env.GetStateSize(), env.GetActionSize(), LoadSavedModels);


        StartCoroutine(Step());
        //Physics.autoSimulation = false;
        //Step();
	}

    IEnumerator Step()
    {
        int currentEpisode = 0;
        int stateSize = env.GetStateSize();
        int actionSize = env.GetActionSize();

        List<float> rewards = new List<float>(MaxTimeSteps);
        List<float> states = new List<float>(MaxTimeSteps * stateSize);
        List<float> actions1 = new List<float>(MaxTimeSteps);
        List<float> actions2 = new List<float>(MaxTimeSteps);

        List<float> advantages = new List<float>(MaxTimeSteps);
        List<float> futureRewards = new List<float>(MaxTimeSteps);

        List<float> rewardsAvg = new List<float>(100);
        List<float> totalRewardsAvg = new List<float>(MaxEpisodes / 100);

        List<float> policyLossAvg = new List<float>(100);
        List<float> totalPolicyLossAvg = new List<float>(MaxEpisodes / 100);

        List<float> valueLossAvg = new List<float>(100);
        List<float> totalValueLossAvg = new List<float>(MaxEpisodes / 100);

        while (currentEpisode < MaxEpisodes)
        {
            env.Reset();
            rewards.Clear();
            states.Clear();
            actions1.Clear();
            actions2.Clear();
            advantages.Clear();
            futureRewards.Clear();

            var state = env.GetState();

            int t = 0;
            for (; t < MaxTimeSteps; ++t)
            {
                var currentActions = m_agent.Act(state, m_device);
                //Debug.Log("Actions: " + currentActions[0] + " " + currentActions[1]);
                env.Act(currentActions[0], currentActions[1]);

                states.AddRange(state);
                actions1.Add(currentActions[0]);
                actions2.Add(currentActions[1]);

                //update Physics
                yield return new WaitForFixedUpdate();
                //Physics.Simulate(0.02f);

                bool isDone = env.IsDone();

                if(isDone)
                {
                    rewards.Add(0.0f);
                    break;
                }

                rewards.Add(Time.fixedDeltaTime);   //reward is 1 per second

                float[] nextState = env.GetState();
                state = nextState;
            }

            for(int i = 0; i < states.Count / stateSize; ++i)
            {
                var currentVal = m_agent.Predict(states.GetRange(i * stateSize, stateSize).ToArray(), m_device);

                float totalRewards = 0.0f;
                for (int j = 0; j < rewards.Count; ++j)
                {
                    totalRewards += rewards[j] * (float)System.Math.Pow(Gamma, (float)j);
                }

                advantages.Add(totalRewards - currentVal);
                futureRewards.Add(totalRewards);
            }

            float rewardsSum = 0.0f;
            for (int j = 0; j < rewards.Count; ++j)
            {
                rewardsSum += rewards[j];
            }

            //Debug.Log("Rewards: " + rewardsSum);

            m_agent.TrainValueModel(states.ToArray(), futureRewards.ToArray(), m_device);
            m_agent.TrainPolicyModel(states.ToArray(), advantages.ToArray(), actions1.ToArray(), actions2.ToArray(), m_device);

            currentEpisode++;

            rewardsAvg.Add(rewardsSum);
            policyLossAvg.Add(m_agent.GetPolicyModelLoss());
            valueLossAvg.Add(m_agent.GetValueModelLoss());

            if (rewardsAvg.Count >= 100)
            {
                ProcessAvg(ref rewardsAvg, ref totalRewardsAvg);
                ProcessAvg(ref policyLossAvg, ref totalPolicyLossAvg);
                ProcessAvg(ref valueLossAvg, ref totalValueLossAvg);

                graph.ShowGraph(totalRewardsAvg, 0);
                graph.ShowGraph(totalPolicyLossAvg, 1);
                graph.ShowGraph(totalValueLossAvg, 2);

                if(SaveTraining)
                {
                    m_agent.SaveModel();
                }

                Debug.Log("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Rewards Avg: " + totalRewardsAvg[totalRewardsAvg.Count - 1]);
            }
        }

        graph.transform.parent.gameObject.SetActive(true);
    }

    private void ProcessAvg(ref List<float> avgList, ref List<float> totalAvgList)
    {
        float avg = 0.0f;
        for (int i = 0; i < avgList.Count; ++i)
        {
            avg += avgList[i];
        }

        avg /= avgList.Count;

        avgList.Clear();

        totalAvgList.Add(avg);
    }

    private DeviceDescriptor m_device;
    private VPGAgent m_agent;
}


////https://github.com/kvfrans/openai-cartpole/blob/master/cartpole-policygradient.py
//public class PGActor
//{
//    const float Epsilon = 0.0001f;
//    const int LayerSize = 64;
//    const float LearningRate = 0.005f;
//    const string SavePolicyModelFileName = "policyModel";
//    const string SaveValueModelFileName = "valueModel";
//    const string SavePolicyTrainerFileName = "policyTrainer";
//    const string SaveValueTrainerFileName = "valueTrainer";

//    public PGActor(int stateSize, int actionSize, bool loadSavedModels)
//    {
//        for (int i = 0; i < actionSize; ++i)
//        {
//            m_normDist.Add(new EDA.Gaussian());
//        }

//        m_inputState = CNTKLib.InputVariable(new int[] { stateSize }, DataType.Float);

//        CreatePolicyModel(stateSize, actionSize);
//        CreateValueModel(stateSize);

//        if (loadSavedModels)
//        {
//            if(File.Exists(SavePolicyModelFileName))
//                m_policyModel.Restore(SavePolicyModelFileName);

//            if (File.Exists(SaveValueModelFileName))
//                m_policyModel.Restore(SaveValueModelFileName);

//            if (File.Exists(SavePolicyTrainerFileName))
//                m_policyTrainer.RestoreFromCheckpoint(SavePolicyTrainerFileName);

//            if (File.Exists(SaveValueTrainerFileName))
//                m_policyTrainer.RestoreFromCheckpoint(SaveValueTrainerFileName);
//        }


//    }

//    private void CreatePolicyModel(int stateSize, int actionSize)
//    {
//        var l1 = CNTKLib.Tanh(Utils.Layer(m_inputState, LayerSize));
//        var l2 = CNTKLib.Tanh(Utils.Layer(l1, LayerSize));

//        var model = new VariableVector();
//        var logProbList = new VariableVector();

//        for (int i = 0; i < actionSize; ++i)
//        {
//            var mean = CNTKLib.Tanh(Utils.Layer(l2, 1));
//            var std = CNTKLib.ReLU(Utils.Layer(l2, 1));
//            m_mean.Add(mean);
//            m_std.Add(std);

//            model.Add(mean);
//            model.Add(std);

//            m_normalSamples.Add(CNTKLib.InputVariable(new int[] { 1 }, DataType.Float));

//            logProbList.Add(Utils.GaussianLogProbabilityLayer(m_mean[i], m_std[i], m_normalSamples[i]));
//        }

//        var combinedLogProb = CNTKLib.Splice(logProbList, new Axis(0));

//        m_policyAdvantage = CNTKLib.InputVariable(new int[] { 1 }, DataType.Float);
//        m_policyLoss = CNTKLib.Negate(CNTKLib.ReduceSum(CNTKLib.ElementTimes(combinedLogProb, m_policyAdvantage), Axis.AllStaticAxes()));
  
//        m_policyModel = CNTKLib.Splice(model, new Axis(0));

//        var learningRate = new TrainingParameterScheduleDouble(LearningRate);
//        var options = new AdditionalLearningOptions();
//        options.gradientClippingThresholdPerSample = 1;
//        options.l2RegularizationWeight = 0.01;
        
//        var learner = new List<Learner>() { Learner.SGDLearner(m_policyModel.Parameters(), learningRate, options) };

//        m_policyTrainer = Trainer.CreateTrainer(m_policyModel, m_policyLoss, m_policyLoss, learner);
//    }

//    private void CreateValueModel(int stateSize)
//    {
//        var l1 = CNTKLib.ReLU(Utils.Layer(m_inputState, LayerSize));
//        var l2 = CNTKLib.ReLU(Utils.Layer(l1, LayerSize));

//        m_valueModel = CNTKLib.ReLU(Utils.Layer(l2, 1));

//        m_valueInput = CNTKLib.InputVariable(new int[] { 1 }, DataType.Float);

//        m_valueLoss = CNTKLib.SquaredError(m_valueModel, m_valueInput);

//        var learningRate = new TrainingParameterScheduleDouble(LearningRate);
//        var options = new AdditionalLearningOptions();
//        options.gradientClippingThresholdPerSample = 1;
//        options.l2RegularizationWeight = 0.01;
//        var learner = new List<Learner>() { Learner.SGDLearner(m_valueModel.Parameters(), learningRate, options) };

//        m_valueTrainer = Trainer.CreateTrainer(m_valueModel, m_valueLoss, m_valueLoss, learner);
//    }

//    public float[] Act(float[] state, DeviceDescriptor device)
//    {
//        Value input = Value.CreateBatch<float>(new int[] { state.Length }, state, device);

//        var inputDict = new Dictionary<Variable, Value>()
//        {
//            { m_inputState, input },
//        };

//        var outputDict = new Dictionary<Variable, Value>()
//        {
//            { m_policyModel.Output, null },
//        };

//        m_policyModel.Evaluate(inputDict, outputDict, device);

//        var outputValue = outputDict[m_policyModel.Output];
//        var denseData = outputValue.GetDenseData<float>(m_policyModel.Output)[0];

//        float[] denseDataFloat = new float[denseData.Count];
//        denseData.CopyTo(denseDataFloat, 0);

//        m_normDist[0].Mean = denseDataFloat[0];
//        m_normDist[0].StdDev = denseDataFloat[1];
//        m_normDist[1].Mean = denseDataFloat[2];
//        m_normDist[1].StdDev = denseDataFloat[3];

//        var sample1 = Mathf.Clamp((float)m_normDist[0].Next(), -1.0f, 1.0f);
//        var sample2 = Mathf.Clamp((float)m_normDist[1].Next(), -1.0f, 1.0f);

//        return new float[2] { sample1, sample2 };
//    }

//    public float Predict(float[] state, DeviceDescriptor device)
//    {
//        Value input = Value.CreateBatch<float>(new int[] { state.Length }, state, device);

//        var inputDict = new Dictionary<Variable, Value>()
//        {
//            { m_inputState, input },
//        };

//        var outputDict = new Dictionary<Variable, Value>()
//        {
//            { m_valueModel.Output, null },
//        };

//        m_valueModel.Evaluate(inputDict, outputDict, device);

//        var outputValue = outputDict[m_valueModel.Output];
//        var denseData = outputValue.GetDenseData<float>(m_valueModel.Output)[0];

//        float[] denseDataFloat = new float[denseData.Count];
//        denseData.CopyTo(denseDataFloat, 0);

//        return UnityEngine.Mathf.Clamp(denseDataFloat[0], -1.0f, 1.0f);
//    }

//    public void TrainValueModel(float[] states, float[] futureRewards, DeviceDescriptor device)
//    {
//        Value input = Value.CreateBatch<float>(new int[] { 9 }, states, device);
//        Value output = Value.CreateBatch<float>(new int[] { 1 }, futureRewards, device);

//        var arguments = new Dictionary<Variable, Value>()
//            {
//                { m_inputState, input },
//                { m_valueInput, output}
//            };

//        m_valueTrainer.TrainMinibatch(arguments, false, device);
//        //Debug.Log("VLoss: " + m_valueTrainer.PreviousMinibatchLossAverage());
//    }

//    public void TrainPolicyModel(float[] states, float[] advantages, float[] actions1, float[] actions2, DeviceDescriptor device)
//    {
//        Value stateData = Value.CreateBatch<float>(new int[] { 9 }, states, device);
//        Value sample1 = Value.CreateBatch<float>(new int[] { 1 }, actions1, device);
//        Value sample2 = Value.CreateBatch<float>(new int[] { 1 }, actions2, device);
//        Value output = Value.CreateBatch<float>(new int[] { 1 }, advantages, device);

//        var arguments = new Dictionary<Variable, Value>()
//            {
//                { m_inputState, stateData },
//                { m_normalSamples[0],  sample1},
//                { m_normalSamples[1],  sample2},
//                { m_policyAdvantage, output}
//            };

//        m_policyTrainer.TrainMinibatch(arguments, false, device);

//        //Debug.Log("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!PLoss2: " + m_policyTrainer.PreviousMinibatchLossAverage());
//    }

//    public float GetPolicyModelLoss()
//    {
//        return (float)m_policyTrainer.PreviousMinibatchLossAverage();
//    }

//    public float GetValueModelLoss()
//    {
//        return (float)m_valueTrainer.PreviousMinibatchLossAverage();
//    }

//    public void SaveModel()
//    {
//        m_policyModel.Save(SavePolicyModelFileName);
//        m_valueModel.Save(SaveValueModelFileName);
//        m_policyTrainer.SaveCheckpoint(SavePolicyTrainerFileName);
//        m_valueTrainer.SaveCheckpoint(SaveValueTrainerFileName);
//    }

//    Function m_policyModel;
//    Function m_valueModel;

//    Variable m_inputState;
//    Variable m_policyAdvantage;

//    private List<Function> m_mean = new List<Function>();
//    private List<Function> m_std = new List<Function>();
//    private List<Variable> m_normalSamples = new List<Variable>();
//    private Function m_policyLoss;
//    private Trainer m_policyTrainer;
//    private Variable m_valueInput;
//    private Function m_valueLoss;
//    private Trainer m_valueTrainer;

//    private List<EDA.Gaussian> m_normDist = new List<EDA.Gaussian>();

//}
