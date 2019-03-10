using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CNTK;

class Agent
{
    public Agent(int stateSize, int actionSize, int layerSize)
    {
        m_stateSize = stateSize;
        m_actionSize = actionSize;

        m_localNetwork = Model.CreateNetwork(m_stateSize, m_actionSize, layerSize);
        //m_targetNetwork = Model.CreateNetwork(m_stateSize, m_actionSize, layerSize);

        m_stateInput = CNTKLib.InputVariable(new int[] { m_stateSize }, DataType.Float, "stateInput");
        m_qTargetOutput = CNTKLib.InputVariable(new int[] { m_actionSize }, DataType.Float, "targetOutput");

        var loss = CNTKLib.ReduceMean(CNTKLib.Square(CNTKLib.Minus(m_localNetwork.Output, m_qTargetOutput)), new Axis(0));
        var meas = CNTKLib.ReduceMean(CNTKLib.Square(CNTKLib.Minus(m_localNetwork.Output, m_qTargetOutput)), new Axis(0));

        var learningRate = new TrainingParameterScheduleDouble(0.0002);
        var options = new AdditionalLearningOptions();
        options.gradientClippingThresholdPerSample = 10;

        //todo: check that m_localNetwork.Parameters() contains parameters for all layers at this point
        var learner = new List<Learner>() { Learner.SGDLearner(m_localNetwork.Parameters(), learningRate, options) };

        m_trainer = Trainer.CreateTrainer(m_localNetwork, loss, meas, learner);

        m_memory = new Memory(m_stateSize);
    }

    public void Train(int sampleSize, float gamma, DeviceDescriptor device)
    {
        float[] samples = m_memory.GetSamples(sampleSize);

        List<float> states = new List<float>();
        List<float> rewards = new List<float>();

        int experienceSize = (m_stateSize * 2) + 2;

        sampleSize = Math.Min(sampleSize, samples.Length / experienceSize);

        for(int i = 0; i < sampleSize; ++i)
        {
            int start = i * experienceSize;
            for(int j = 0; j < m_stateSize; ++j)
            {
                states.Add(samples[start + j]);
            }

            var reward = samples[start + m_stateSize + 1];  //state size + action + reward offset

            var qValues = GetQValues(states.GetRange(states.Count - 1 - m_stateSize, m_stateSize).ToArray(), device);

            //TODO: if not done, do this, else skip
            reward += gamma * GetMaxReward(qValues);

            rewards.Add(reward);
        }


        float[] statesFlattened = states.ToArray();
        float[] rewardsFlattened = rewards.ToArray();

        //TODO: check that states is working
        Value input = Value.CreateBatch<float>(new int[] { m_stateSize }, statesFlattened, device);
        Value output = Value.CreateBatch<float>(new int[] { 1 }, rewardsFlattened, device);

        var arguments = new Dictionary<Variable, Value>()
        {
            { m_stateInput, input },
            { m_qTargetOutput, output}
        };

        m_trainer.TrainMinibatch(arguments, false, device);
    }

    public void Observe(float[] state, float action, float reward, float[] nextState)
    {
        List<float> experience = new List<float>(state);
        experience.Add(action);
        experience.Add(reward);
        experience.AddRange(nextState);

        m_memory.Add(experience.ToArray());
    }

    public void CreateBatch(int sampleSize)
    {
        //todo: reference memory


    }

    public int Act(float[] state, float epsillon, int actionSize, DeviceDescriptor device)
    {
        var randomNum = UnityEngine.Random.Range(0.0f, 1.0f);

        if(randomNum < epsillon)
        {
            return UnityEngine.Random.Range(0, actionSize);
        }

        var qValues = GetQValues(state, device);

        var action = GetArgMax(qValues);

        return action;
    }

    private IList<float> GetQValues(float[] state, DeviceDescriptor device)
    {
        Value input = Value.CreateBatch<float>(new int[] { state.Length }, state, device);

        var inputDict = new Dictionary<Variable, Value>()
        {
            { m_stateInput, input },
        };

        var outputDict = new Dictionary<Variable, Value>()
        {
            { m_qTargetOutput, null}
        };


        m_localNetwork.Evaluate(inputDict, outputDict, device);

        var outputValue = outputDict[m_qTargetOutput];

        return outputValue.GetDenseData<float>(m_qTargetOutput)[0];
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
                argmaxArray[i] = value;
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
                argmaxArray[i] = value;
            }
        }

        return value;
    }


    private Memory m_memory;

    private Function m_localNetwork;
    //private Function m_targetNetwork;
    private Trainer m_trainer;
    private Variable m_stateInput;
    private Variable m_qTargetOutput;

    private int m_stateSize;
    private int m_actionSize;
}

class ReplayBuffer
{

}

