using CNTK;
using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CNTKTest : MonoBehaviour {

    const int EpisodeCount = 300;
    const int MaxSteps = 20;
    const float MinEpsillon = 0.05f;
    const int LayerSize = 32;
    const float Gamma = 0.95f;
    const int BatchSize = 64;

	// Use this for initialization
	void Start ()
    {
        m_device = DeviceDescriptor.UseDefaultDevice();
        print($"Hello from CNTK for {m_device.Type} only!");

        var env = new Environment();
        var stateSize = env.GetStateSize();
        var actionSize = env.GetActionSize();
        Agent agent = new Agent(stateSize, actionSize, LayerSize);

        Play(agent, env, EpisodeCount, MinEpsillon, true);  //training
        Play(agent, env, 100, 0.0f, false);     //no random and trained
        Play(agent, env, 100, 1.0f, false);     //random
    }

    void Play(Agent agent, Environment env, int episodeCount = 100, float minEpsillon = 0.05f, bool isTraining = true)
    {
        var actionSize = env.GetActionSize();
        var epsillon = isTraining ? 1.0f : minEpsillon;

        var rewardQueue = new Queue<float>(100);

        for (int epi = 0; epi < episodeCount; ++epi)
        {
            env.Reset();

            if(isTraining)
            {
                epsillon = Mathf.Max(minEpsillon, 1.0f - ((float)epi / episodeCount));
            }

            float episodeReward = 0.0f;
            var currentState = Array.ConvertAll<int, float>(env.GetCurrentState(), x => Convert.ToSingle(x));

            List<int> actions = new List<int>(MaxSteps);

            int t = 0;
            for (t = 0; t < MaxSteps; t++)
            {
                var action = agent.Act(currentState, epsillon, actionSize, m_device);

                actions.Add(action);

                float reward = 0.0f;
                bool isFinished = env.Act((Environment.Actions)action, out reward);

                episodeReward += reward;

                var nextState = Array.ConvertAll<int, float>(env.GetCurrentState(), x => Convert.ToSingle(x));

                if(isTraining)
                {
                    agent.Observe(currentState, (float)action, reward, nextState, isFinished ? 1.0f : 0.0f);
                    agent.Train(BatchSize, Gamma, m_device);
                }

                if (isFinished)
                {
                    if(!isTraining)
                    {
                        string path = "";
                        for(int a = 0; a < actions.Count; ++a)
                        {
                            path += " " + actions[a].ToString();
                        }

                        Debug.Log("Path: " + path);
                    }

                    break;
                }

                currentState = nextState;
            }

            rewardQueue.Enqueue((float)t);
            if ((epi + 1) % 100 == 0)
            {
                float rewardAvg = 0.0f;

                while (rewardQueue.Count > 0)
                {
                    rewardAvg += rewardQueue.Dequeue();
                }

                rewardAvg /= 100.0f;

                print("Reward: " + epi + " " + rewardAvg + " " + isTraining);
                print("Loss: " + agent.GetTrainingLoss());
            }
        }
    }

    private DeviceDescriptor m_device;
    
}


