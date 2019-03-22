using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CNTK;

class Agent
{
    const double LearningRate = 0.01;
    const float TAU = 0.00005f;

    public Agent(int stateSize, int actionSize, int layerSize)
    {
        m_stateSize = stateSize;
        m_actionSize = actionSize;

        m_localNetwork = Model.CreateNetwork(m_stateSize, m_actionSize, layerSize, out m_stateInput);
        m_targetNetwork = Model.CreateNetwork(m_stateSize, m_actionSize, layerSize, out m_stateTargetInput);

        m_qTargetOutput = CNTKLib.InputVariable(new int[] { m_actionSize }, DataType.Float, "targetOutput");

        var loss = CNTKLib.ReduceMean(CNTKLib.Square(CNTKLib.Minus(m_localNetwork.Output, m_qTargetOutput)), new Axis(0));
        var meas = CNTKLib.ReduceMean(CNTKLib.Square(CNTKLib.Minus(m_localNetwork.Output, m_qTargetOutput)), new Axis(0));

        var vp = new VectorPairSizeTDouble()
        {
            new PairSizeTDouble(1, 0.1),
            new PairSizeTDouble(1, 0.05),
            new PairSizeTDouble(1, 0.02),
            new PairSizeTDouble(1, 0.01),
            new PairSizeTDouble(1, 0.005),
        };

        var learningRate = new TrainingParameterScheduleDouble(vp, 400);

        var learner = new List<Learner>() { Learner.SGDLearner(m_localNetwork.Parameters(), learningRate) };

        m_trainer = Trainer.CreateTrainer(m_localNetwork, loss, meas, learner);

        m_memory = new Memory(m_stateSize, 512);
    }

    public void Train(int sampleSize, float gamma, DeviceDescriptor device)
    {
        int[] indexes;
        float[] samples = m_memory.GetSamples(sampleSize, out indexes);

        int experienceSize = (m_stateSize * 2) + 3;

        var currentSampleCount = samples.Length / experienceSize;

        if(currentSampleCount < sampleSize)
        {
            return;
        }

        List<float> states = new List<float>(m_stateSize * sampleSize);
        List<float> rewards = new List<float>(m_actionSize * sampleSize);
        List<float> errors = new List<float>(sampleSize);

        for (int i = 0; i < sampleSize; ++i)
        {
            int start = i * experienceSize;
            for(int j = 0; j < m_stateSize; ++j)
            {
                states.Add(samples[start + j]);
            }

            float[] exp = new float[experienceSize];
            Array.Copy(samples, start, exp, 0, experienceSize);
            float error;
            var qValues = CalculateQValues(exp, gamma, device, out error);

            rewards.AddRange(qValues);
            errors.Add(error);
        }

        for(int i = 0; i < sampleSize; ++i)
        {
            m_memory.Update(indexes[i], errors[i]);
        }

        float[] statesFlattened = states.ToArray();
        float[] rewardsFlattened = rewards.ToArray();

        Value input = Value.CreateBatch<float>(new int[] { m_stateSize }, statesFlattened, device);
        Value output = Value.CreateBatch<float>(new int[] { m_actionSize }, rewardsFlattened, device);

        var arguments = new Dictionary<Variable, Value>()
        {
            { m_stateInput, input },
            { m_qTargetOutput, output}
        };

        m_trainer.TrainMinibatch(arguments, false, device);

        //Model.SoftUpdate(m_localNetwork, m_targetNetwork, device, TAU);
    }

    public void TransferLearning(DeviceDescriptor device)
    {
        Model.SoftUpdate(m_localNetwork, m_targetNetwork, device, 1.0f);
    }

    public void Observe(float[] state, float action, float reward, float[] nextState, float isDone, float gamma, DeviceDescriptor device)
    {
        List<float> experience = new List<float>(state);
        experience.Add(action);
        experience.Add(reward);
        experience.AddRange(nextState);
        experience.Add(isDone);

        var experienceArray = experience.ToArray();
        float error;
        CalculateQValues(experienceArray, gamma, device, out error);

        m_memory.Add(error, experienceArray);
    }

    public float[] CalculateQValues(float[] experience, float gamma, DeviceDescriptor device, out float error)
    {
        float[] currentState = new float[m_stateSize];
        Array.Copy(experience, 0, currentState, 0, m_stateSize);

        var action = (int)experience[m_stateSize];
        var reward = experience[m_stateSize + 1];  //state size + action + reward offset
        var isDone = experience[(m_stateSize * 2) + 2] > 0.0f;

        var qValues = GetLocalQValues(currentState, device).ToArray<float>();

        var prevVal = qValues[action];
        qValues[action] = reward;

        if (!isDone)
        {
            float[] nextState = new float[m_stateSize];
            Array.Copy(experience, m_stateSize + 2, nextState, 0, m_stateSize);

            var ddqnAction = GetArgMax(GetLocalQValues(nextState, device));

            qValues[action] += gamma * GetTargetQValues(nextState, device)[ddqnAction];
        }

        qValues[action] = UnityEngine.Mathf.Clamp(qValues[action], -1.0f, 1.0f);

        error = Math.Abs(qValues[action] - prevVal);

        return qValues;
    }

    public int Act(float[] state, float epsillon, int actionSize, DeviceDescriptor device, bool useTargetNetwork = false)
    {
        var randomNum = UnityEngine.Random.Range(0.0f, 1.0f);

        if(randomNum < epsillon)
        {
            return UnityEngine.Random.Range(0, actionSize);
        }

        var qValues = GetLocalQValues(state, device);

        if(useTargetNetwork)
        {
            qValues = GetTargetQValues(state, device);
        }

        var action = GetArgMax(qValues);

        return action;
    }

    public IList<float> GetLocalQValues(float[] state, DeviceDescriptor device)
    {
        Value input = Value.CreateBatch<float>(new int[] { state.Length }, state, device);

        var inputDict = new Dictionary<Variable, Value>()
        {
            { m_stateInput, input },
        };

        var outputDict = new Dictionary<Variable, Value>()
        {
            {m_localNetwork.Output, null }
        };

        m_localNetwork.Evaluate(inputDict, outputDict, device);

        var outputValue = outputDict[m_localNetwork.Output];

        return outputValue.GetDenseData<float>(m_localNetwork.Output)[0];
    }

    public IList<float> GetTargetQValues(float[] state, DeviceDescriptor device)
    {
        Value input = Value.CreateBatch<float>(new int[] { state.Length }, state, device);

        var inputDict = new Dictionary<Variable, Value>()
        {
            { m_stateTargetInput, input },
        };

        var outputDict = new Dictionary<Variable, Value>()
        {
            {m_targetNetwork.Output, null }
        };


        m_targetNetwork.Evaluate(inputDict, outputDict, device);

        var outputValue = outputDict[m_targetNetwork.Output];

        return outputValue.GetDenseData<float>(m_targetNetwork.Output)[0];
    }

    private int GetArgMax(IList<float> argmaxArray)
    {
        Debug.Assert(argmaxArray.Count > 0);

        float value = argmaxArray[0];
        int index = 0;

        for(int i = 1; i < argmaxArray.Count; ++i)
        {
            if(argmaxArray[i] > value)
            {
                value = argmaxArray[i];
                index = i;
            }
        }

        return index;
    }

    private float GetMaxReward(IList<float> argmaxArray)
    {
        Debug.Assert(argmaxArray.Count > 0);

        float value = argmaxArray[0];

        for (int i = 1; i < argmaxArray.Count; ++i)
        {
            if (argmaxArray[i] > value)
            {
                value = argmaxArray[i];
            }
        }

        return value;
    }

    public double GetTrainingLoss()
    {
        return m_trainer.PreviousMinibatchLossAverage();
    }


    private Memory m_memory;

    private Function m_localNetwork;
    private Function m_targetNetwork;

    private Trainer m_trainer;
    private Variable m_stateInput;
    private Variable m_stateTargetInput;
    private Variable m_qTargetOutput;

    private int m_stateSize;
    private int m_actionSize;
}

