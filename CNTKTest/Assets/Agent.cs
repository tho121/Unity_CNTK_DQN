﻿using System;
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

        m_localNetwork = Model.CreateNetwork(m_stateSize, m_actionSize, layerSize, out m_stateInput);
        //m_targetNetwork = Model.CreateNetwork(m_stateSize, m_actionSize, layerSize);

        //TODO: where the fuck does this hook up?
        //m_stateInput = CNTKLib.InputVariable(new int[] { m_stateSize }, DataType.Float, "stateInput");
        m_qTargetOutput = CNTKLib.InputVariable(new int[] { m_actionSize }, DataType.Float, "targetOutput");

        m_lossFunction = CNTKLib.ReduceMean(CNTKLib.Square(CNTKLib.Minus(m_localNetwork.Output, m_qTargetOutput)), new Axis(0));
        var meas = CNTKLib.ReduceMean(CNTKLib.Square(CNTKLib.Minus(m_localNetwork.Output, m_qTargetOutput)), new Axis(0));

        var learningRate = new TrainingParameterScheduleDouble(0.002);
        var options = new AdditionalLearningOptions();
        options.gradientClippingThresholdPerSample = 10;

        //todo: check that m_localNetwork.Parameters() contains parameters for all layers at this point
        var learner = new List<Learner>() { Learner.SGDLearner(m_localNetwork.Parameters(), learningRate, options) };

        m_trainer = Trainer.CreateTrainer(m_localNetwork, m_lossFunction, meas, learner);

        m_memory = new Memory(m_stateSize);
    }

    public void Train(int sampleSize, float gamma, DeviceDescriptor device)
    {
        float[] samples = m_memory.GetSamples(sampleSize);

        int experienceSize = (m_stateSize * 2) + 3;

        var currentSampleCount = samples.Length / experienceSize;

        if(currentSampleCount < sampleSize)
        {
            return;
        }

        List<float> states = new List<float>(m_stateSize * sampleSize);
        List<float> rewards = new List<float>(m_actionSize * sampleSize);
        List<float> actions = new List<float>(sampleSize);

        for (int i = 0; i < sampleSize; ++i)
        {
            int start = i * experienceSize;
            for(int j = 0; j < m_stateSize; ++j)
            {
                states.Add(samples[start + j]);
            }

            //s,a,r,s',done
            var currentState = states.GetRange(states.Count - m_stateSize, m_stateSize).ToArray();
            var action = (int)samples[start + m_stateSize];
            var reward = samples[start + m_stateSize + 1];  //state size + action + reward offset
            var isDone = samples[start + (m_stateSize * 2) + 2] > 0.0f;

            actions.Add(action);

            var qValues = GetQValues(currentState, device);

            qValues[action] = reward;

            if (!isDone)
            {
                var nextState = new List<float>(m_stateSize);

                int nextStateStart = start + m_stateSize + 2;
                for (int j = 0; j < m_stateSize; ++j)
                {
                    nextState.Add(samples[nextStateStart + j]);
                }

                qValues[action] += gamma * GetMaxReward(GetQValues(nextState.ToArray(), device));
            }

            rewards.AddRange(qValues);
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
    }

    public void Observe(float[] state, float action, float reward, float[] nextState, float isDone)
    {
        List<float> experience = new List<float>(state);
        experience.Add(action);
        experience.Add(reward);
        experience.AddRange(nextState);
        experience.Add(isDone);

        m_memory.Add(experience.ToArray());
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
            //{ m_qTargetOutput, null}
            {m_localNetwork.Output, null }
        };


        m_localNetwork.Evaluate(inputDict, outputDict, device);

        var outputValue = outputDict[m_localNetwork.Output];

        return outputValue.GetDenseData<float>(m_localNetwork.Output)[0];
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
                argmaxArray[i] = value;
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
    private Function m_lossFunction;

    private Trainer m_trainer;
    private Variable m_stateInput;
    private Variable m_qTargetOutput;

    private int m_stateSize;
    private int m_actionSize;
}

