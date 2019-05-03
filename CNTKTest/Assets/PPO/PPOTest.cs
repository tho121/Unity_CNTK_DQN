using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using CNTK;
using System;

public class PPOTest : MonoBehaviour {

    public BallEnvironment env;
    
    public int MaxEpisodes = 1000;
    public int MaxTimeSteps = 5000;  //at fixed timestep 0.02 -> 5000 * 0.02 = 100 seconds
    public const float Gamma = 0.99f;
    
    public GraphUI graph;

    // Use this for initialization
    void Start () {

        env.selfReseting = false;

        m_device = DeviceDescriptor.UseDefaultDevice();
        print($"Hello from CNTK for {m_device.Type} only!");

        m_agent = new PPOAgent(env.GetStateSize(), env.GetActionSize(), MaxTimeSteps, m_device);

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
                    rewards.Add(0.0f);
                    break;
                }

                rewards.Add(Time.fixedDeltaTime);   //reward is 1 per second

                state = nextState;
            }

            advantages.AddRange(Utils.CalculateGAE(rewards.ToArray(), targetValues.ToArray(), Gamma));

            for (int i = 0; i < states.Count / stateSize; ++i)
            {
                var currentState = states.GetRange(i * stateSize, stateSize).ToArray();
                var currentActions = actions.GetRange(i * actionSize, actionSize).ToArray();
                var currentProbabilities = probabilities.GetRange(i * actionSize, actionSize).ToArray();

                m_agent.AddExperience(currentState, currentActions, currentProbabilities, targetValues[i], advantages[i]);
            }

            float rewardsSum = 0.0f;
            for (int j = 0; j < rewards.Count; ++j)
            {
                rewardsSum += rewards[j];
            }


            //Debug.Log("Rewards: " + rewardsSum);

            m_agent.Train(m_device);

            currentEpisode++;

            rewardsAvg.Add(rewardsSum);
            policyLossAvg.Add(m_agent.GetModelLoss());
            //valueLossAvg.Add(m_agent.GetValueModelLoss());

            if (rewardsAvg.Count >= 100)
            {
                ProcessAvg(ref rewardsAvg, ref totalRewardsAvg);
                ProcessAvg(ref policyLossAvg, ref totalPolicyLossAvg);
                //ProcessAvg(ref valueLossAvg, ref totalValueLossAvg);

                graph.ShowGraph(totalRewardsAvg, 0);
                graph.ShowGraph(totalPolicyLossAvg, 1);

                Debug.Log("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Rewards Avg: " + totalRewardsAvg[totalRewardsAvg.Count - 1]);
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
}
