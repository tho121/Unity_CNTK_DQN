using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CNTK;

namespace DQN
{
    class Agent
    {
        public Agent(int stateSize, int actionSize, int layerSize)
        {
            m_stateSize = stateSize;
            m_actionSize = actionSize;

            m_localNetwork = Model.CreateNetwork(m_stateSize, m_actionSize, layerSize, out m_stateInput);
            m_targetNetwork = Model.CreateNetwork(m_stateSize, m_actionSize, layerSize, out m_stateTargetInput);

            m_qTargetOutput = CNTKLib.InputVariable(new int[] { m_actionSize }, DataType.Float, "targetOutput");

            var loss = CNTKLib.Square(CNTKLib.Minus(m_localNetwork, m_qTargetOutput));
            var meas = CNTKLib.Square(CNTKLib.Minus(m_localNetwork, m_qTargetOutput));

            //learning rate schedule
            var vp = new VectorPairSizeTDouble()
            {
                //new PairSizeTDouble(2, 0.2),
                //new PairSizeTDouble(1, 0.1),
                //new PairSizeTDouble(1, 0.05),
                //new PairSizeTDouble(1, 0.02),
                new PairSizeTDouble(1, 0.02),
                new PairSizeTDouble(1, 0.01),
            };

            //per training batch
            var learningRate = new TrainingParameterScheduleDouble(vp, 4000);

            var learner = new List<Learner>() { Learner.SGDLearner(m_localNetwork.Parameters(), learningRate) };

            m_trainer = Trainer.CreateTrainer(m_localNetwork, loss, null, learner);

            m_memory = new Memory((stateSize * 2) + 3);//current state, action, reward, next state, done
        }

        public void Train(int sampleSize, float gamma, DeviceDescriptor device)
        {
            float[] samples = m_memory.GetSamples(sampleSize);

            int experienceSize = (m_stateSize * 2) + 3;

            var currentSampleCount = samples.Length / experienceSize;

            if (currentSampleCount < sampleSize)
            {
                return;
            }

            List<float> states = new List<float>(m_stateSize * sampleSize);
            List<float> rewards = new List<float>(m_actionSize * sampleSize);

            for (int i = 0; i < sampleSize; ++i)
            {
                int start = i * experienceSize;
                for (int j = 0; j < m_stateSize; ++j)
                {
                    states.Add(samples[start + j]);
                }

                //s,a,r,s',done
                var currentState = states.GetRange(states.Count - m_stateSize, m_stateSize).ToArray();
                var action = (int)samples[start + m_stateSize];
                var reward = samples[start + m_stateSize + 1];  //state size + action + reward offset
                var isDone = samples[start + (m_stateSize * 2) + 2] > 0.0f;

                var qValues = GetLocalQValues(currentState, device).ToArray<float>();

                qValues[action] = reward;

                if (!isDone)
                {
                    var nextState = new List<float>(m_stateSize);

                    int nextStateStart = start + m_stateSize + 2;
                    for (int j = 0; j < m_stateSize; ++j)
                    {
                        nextState.Add(samples[nextStateStart + j]);
                    }

                    qValues[action] += gamma * GetMaxReward(GetTargetQValues(nextState.ToArray(), device));
                }

                qValues[action] = UnityEngine.Mathf.Clamp(qValues[action], -1.0f, 1.0f);

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

        public void TransferLearning(DeviceDescriptor device)
        {
            Model.SoftUpdate(m_localNetwork, m_targetNetwork, device, 1.0f);
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

        public int Act(float[] state, float epsillon, int actionSize, DeviceDescriptor device, bool useTargetNetwork = false)
        {
            var randomNum = UnityEngine.Random.Range(0.0f, 1.0f);

            if (randomNum < epsillon)
            {
                return UnityEngine.Random.Range(0, actionSize);
            }

            var qValues = GetLocalQValues(state, device);

            if (useTargetNetwork)
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

        private IList<float> GetTargetQValues(float[] state, DeviceDescriptor device)
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

            for (int i = 1; i < argmaxArray.Count; ++i)
            {
                if (argmaxArray[i] > value)
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
}

