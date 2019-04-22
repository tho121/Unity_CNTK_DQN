using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using CNTK;
using System;

public class PPOTest : MonoBehaviour {

    public VPGEnvironment env;
    
    public const int MaxEpisodes = 80000;
    public const int MaxTimeSteps = 10000;  //at fixed timestep 0.02 -> 200 seconds
    public const float Gamma = 0.9555f;
    
    public GraphUI graph;

    // Use this for initialization
    void Start () {

        env.selfReseting = false;

        m_device = DeviceDescriptor.UseDefaultDevice();
        print($"Hello from CNTK for {m_device.Type} only!");

        m_agent = new PGActor(env.GetStateSize(), env.GetActionSize());


        StartCoroutine(Step());
	}
	
	// Update is called once per frame
	//void Update () {
		
	//}

 //   private void FixedUpdate()
 //   {
        
 //   }

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

                Debug.Log("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Rewards Avg: " + totalRewardsAvg[totalRewardsAvg.Count - 1]);
            }
        }

        graph.transform.parent.gameObject.SetActive(true);
        graph.ShowGraph(totalRewardsAvg, 0);
        graph.ShowGraph(totalPolicyLossAvg, 1);
        graph.ShowGraph(totalValueLossAvg, 2);
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
    private PGActor m_agent;
}
