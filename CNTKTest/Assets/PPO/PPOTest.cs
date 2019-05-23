using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using CNTK;
using System;

public class PPOTest : MonoBehaviour {

    public BallEnvironment env;
    
    public int MaxEpisodes = 1000;
    public int MaxTimeSteps = 5000;  //at fixed timestep 0.02 -> 5000 * 0.02 = 100 seconds

    public const float Gamma = 0.995f;
    public const float Lambda = 0.95f;

    public bool LoadSavedModel = true;
    public bool SaveTraining = true;

    public bool SaveNow = false;
    public bool DebugLosses = false;

    const int MiniBatchSize = 1024;
    const int Epochs = 5;
    
    public GraphUI graph;

    // Use this for initialization
    void Start () {

        env.selfReseting = false;

        m_device = DeviceDescriptor.UseDefaultDevice();
        print($"Hello from CNTK for {m_device.Type} only!");

        m_agent = new PPOAgent(env.GetStateSize(), env.GetActionSize(), MaxTimeSteps, m_device, LoadSavedModel, DebugLosses);

        StartCoroutine(Step());
	}
	
    IEnumerator Step()
    {
        int currentEpisode = 0;
        int stateSize = env.GetStateSize();
        int actionSize = env.GetActionSize();

        List<float> rewards = new List<float>(MaxTimeSteps);
        List<float> states = new List<float>(MaxTimeSteps * stateSize);
        List<float> actions = new List<float>(MaxTimeSteps * actionSize);

        List<float> probabilities = new List<float>(MaxTimeSteps * actionSize);
        List<float> advantages = new List<float>(MaxTimeSteps);
        List<float> futureRewards = new List<float>(MaxTimeSteps);
        List<float> targetValues = new List<float>(MaxTimeSteps);

        List<float> rewardsAvg = new List<float>(100);
        List<float> totalRewardsAvg = new List<float>(MaxEpisodes * Epochs / 100);

        List<float> policyLossAvg = new List<float>(100);
        List<float> totalPolicyLossAvg = new List<float>(MaxEpisodes * Epochs / 100);

        List<float> valueLossAvg = new List<float>(100);
        List<float> totalValueLossAvg = new List<float>(MaxEpisodes * Epochs / 100);

        List<float> entropyLossAvg = new List<float>(100);
        List<float> totalEntropyLossAvg = new List<float>(MaxEpisodes * Epochs / 100);

        List<float> totalLossAvg = new List<float>(100);
        List<float> totalModelLossAvg = new List<float>(MaxEpisodes * Epochs / 100);

        while (currentEpisode < MaxEpisodes)
        {
            env.Reset();
            rewards.Clear();
            states.Clear();
            actions.Clear();
            advantages.Clear();
            futureRewards.Clear();
            targetValues.Clear();
            probabilities.Clear();

            var state = env.GetState();

            int t = 0;
            for (; t < MaxTimeSteps; ++t)
            {
                var currentProbabilites = new float[actionSize];
                var currentActions = m_agent.Act(state, m_device, out currentProbabilites);
                //Debug.Log("Actions: " + currentActions[0] + " " + currentActions[1]);
                env.Act(currentActions[0], currentActions[1]);

                states.AddRange(state);
                actions.AddRange(currentActions);
                probabilities.AddRange(currentProbabilites);
                targetValues.Add(m_agent.Predict(state, m_device));
                //update Physics
                yield return new WaitForFixedUpdate();

                bool isDone = env.IsDone();

                float[] nextState = env.GetState();

                if (isDone)
                {
                    rewards.Add(-1.0f);
                    break;
                }

                rewards.Add(Time.fixedDeltaTime);   //reward is 1 per second

                state = nextState;
            }

            float rewardsSum = 0.0f;
            for (int j = 0; j < rewards.Count; ++j)
            {
                rewardsSum += rewards[j];
            }

            advantages.AddRange(Utils.GeneralAdvantageEst(rewards.ToArray(), targetValues.ToArray(), Gamma, Lambda, -1.0f));

            //for (int i = rewards.Count - 2; i >= 0 ; --i)
            //{
            //    rewards[i] = rewards[i] + rewards[i + 1] * Gamma;
            //}

            //double[] advantagesConverted = new double[rewards.Count];

            for (int i = 0; i < advantages.Count; ++i)
            {
                //var adv = rewards[i] - targetValues[i];
                //advantages.Add(adv);

                targetValues[i] += advantages[i];

                //advantagesConverted[i] = (double)advantages[i];
            }



            //m_normalDist.Process(advantagesConverted);

            //for (int i = 0; i < advantages.Count; ++i)
            //{
            //    //advantages[i] -= targetValues[i];
            //    advantages[i] = (advantages[i] - (float)m_normalDist.Mean) / ((float)m_normalDist.StdDev + 0.0000001f);
            //}



            //advantages.AddRange(Utils.CalculateGAE(rewards.ToArray(), targetValues.ToArray(), Gamma));

            for (int i = 0; i < states.Count / stateSize; ++i)
            {
                var currentState = states.GetRange(i * stateSize, stateSize).ToArray();
                var currentActions = actions.GetRange(i * actionSize, actionSize).ToArray();
                var currentProbabilities = probabilities.GetRange(i * actionSize, actionSize).ToArray();

                m_agent.AddExperience(currentState, currentActions, currentProbabilities, targetValues[i], advantages[i]);
            }

            //Debug.Log("Rewards: " + rewardsSum);

            //m_agent.Train(m_device);

            currentEpisode++;

            rewardsAvg.Add(rewardsSum);
            //policyLossAvg.Add(m_agent.GetModelLoss());
            //valueLossAvg.Add(m_agent.GetValueModelLoss());

            if(m_agent.GetMemory().GetCurrentMemorySize() >= MiniBatchSize)
            {
                for(int i = 0; i < Epochs; ++i)
                {
                    m_agent.Train(m_device);

                    if(DebugLosses)
                    {
                        var losses = m_agent.GetCurrentLoss();

                        policyLossAvg.Add(losses[0]);
                        valueLossAvg.Add(losses[1]);
                        entropyLossAvg.Add(losses[2]);
                    }

                    totalLossAvg.Add(m_agent.GetModelLoss());

                }

                m_agent.GetMemory().ClearMemory();

                if(DebugLosses)
                {
                    ProcessAvg(ref policyLossAvg, ref totalPolicyLossAvg);
                    ProcessAvg(ref valueLossAvg, ref totalValueLossAvg);
                    ProcessAvg(ref entropyLossAvg, ref totalEntropyLossAvg);
                }
                
                ProcessAvg(ref totalLossAvg, ref totalModelLossAvg);
            }


            if (rewardsAvg.Count >= 100)
            {
                ProcessAvg(ref rewardsAvg, ref totalRewardsAvg);
                //ProcessAvg(ref policyLossAvg, ref totalPolicyLossAvg);
                //ProcessAvg(ref valueLossAvg, ref totalValueLossAvg);

                graph.ShowGraph(totalRewardsAvg, 0);
                graph.ShowGraph(totalModelLossAvg , 1);

                if (DebugLosses)
                {
                    graph.ShowGraph(totalValueLossAvg, 2);
                    graph.ShowGraph(totalEntropyLossAvg, 3);
                    graph.ShowGraph(totalPolicyLossAvg, 4);
                }

                //var losses = m_agent.GetCurrentLoss();
                //Debug.Log(losses[0] + " " + losses[1] + " " + losses[2]);
                Debug.Log("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Rewards Avg: " + totalRewardsAvg[totalRewardsAvg.Count - 1] + " " + currentEpisode);
            }

            if(SaveNow)
            {
                SaveNow = false;
                m_agent.SaveModel();
            }

            if(SaveTraining && (currentEpisode + 1) % 5000 == 0)
            {
                m_agent.SaveModel();
            }
        }

        graph.transform.parent.gameObject.SetActive(true);
        
        //graph.ShowGraph(totalValueLossAvg, 2);
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
    private PPOAgent m_agent;
    private EDA.Gaussian m_normalDist = new EDA.Gaussian();
}
