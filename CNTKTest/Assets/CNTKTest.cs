using CNTK;
using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CNTKTest : MonoBehaviour {

    public GraphUI graphScript;

    const int EpisodeCount = 2000;
    const int MaxSteps = 10;
    const float MinEpsillon = 0.01f;
    const int LayerSize = 64;
    const float Gamma = 0.9f;
    const int BatchSize = 64;

    const int PrintInterval = 100;

    const float Lambda = 0.005f;    //approaches 2000

	// Use this for initialization
	void Start ()
    {
        m_device = DeviceDescriptor.UseDefaultDevice();
        print($"Hello from CNTK for {m_device.Type} only!");

        m_allRewards = new List<float>(EpisodeCount);

        m_environment = new Environment();
        var stateSize = m_environment.GetStateSize();
        var actionSize = m_environment.GetActionSize();
        m_agent = new Agent(stateSize, actionSize, LayerSize);

        m_currentCoroutine = StartCoroutine(Play(m_agent, m_environment, EpisodeCount, MinEpsillon, true));  //training
    }

    private void Update()
    {
        if (m_currentCoroutine == null)
        {
            if (m_coroutineCount == 1)
            {
                m_currentCoroutine = StartCoroutine(Play(m_agent, m_environment, 100, 0.0f, false));     //no random and trained
            }
            else if (m_coroutineCount == 2)
            {
                m_agent.TransferLearning(m_device);
                m_currentCoroutine = StartCoroutine(Play(m_agent, m_environment, 100, 0.0f, false));     //no random and trained
            }
            else if (m_coroutineCount == 3)
            {
                m_currentCoroutine = StartCoroutine(Play(m_agent, m_environment, 100, 1.0f, false));     //random
            }
        }
    }

    IEnumerator Play(Agent agent, Environment env, int episodeCount = 100, float minEpsilon = 0.05f, bool isTraining = true)
    {
        var actionSize = env.GetActionSize();
        var epsilon = isTraining ? 1.0f : minEpsilon;

        var rewardQueue = new Queue<float>(100);

        for (int epi = 0; epi < episodeCount; ++epi)
        {
            env.Reset();

            if(isTraining)
            {
                //epsilon = 1.0f;
                //epsilon = Mathf.Max(minEpsilon, 1.0f - (epi * 0.5f /episodeCount));

                //epsilon = minEpsilon + ((1.0f - minEpsilon) * (float)Math.Exp(-Lambda * epi));

                var epsi = Mathf.Max(0.0f, (float)Math.Pow(((float)epi / episodeCount), 4.0f));
                epsilon = Mathf.Max(minEpsilon, 1.0f - epsi);
            }

            float episodeReward = 0.0f;
            var currentState = Array.ConvertAll<int, float>(env.GetCurrentState(), x => Convert.ToSingle(x));

            List<int> actions = new List<int>(MaxSteps);

            int t = 0;
            for (t = 0; t < MaxSteps; t++)
            {
                //debug!
                if(isTraining && t == 0 && (epi % 500 == 0))
                {
                    foreach(var q in agent.GetLocalQValues(currentState, m_device))
                    {
                        print("QVAL: " + epi + " " + q);
                    }

                    foreach (var q in agent.GetTargetQValues(currentState, m_device))
                    {
                        print("TARGET QVAL: " + epi + " " + q);
                    }
                }

                var action = agent.Act(currentState, epsilon, actionSize, m_device, !isTraining);

                actions.Add(action);

                float reward = 0.0f;
                bool isFinished = env.Act((Environment.Actions)action, out reward);

                episodeReward += reward;

                var nextState = Array.ConvertAll<int, float>(env.GetCurrentState(), x => Convert.ToSingle(x));

                if(isTraining)
                {
                    agent.Observe(currentState, (float)action, reward, nextState, isFinished ? 1.0f : 0.0f, Gamma, m_device);
                    agent.Train(BatchSize, Gamma, m_device);
                }

                if (isFinished)
                {
                    if(false)//!isTraining)
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

            if(isTraining)
            {
                //agent.Train(BatchSize, Gamma, m_device);

                if(epi > BatchSize)
                {
                    m_allRewards.Add((float)agent.GetTrainingLoss());

                    if (graphScript != null)
                    {
                        graphScript.ShowGraph(m_allRewards);
                    }

                }

                if ((epi + 1) % 50 == 0)
                {
                    agent.TransferLearning(m_device);
                }
            }

            rewardQueue.Enqueue((float)episodeReward);
            if ((epi + 1) % PrintInterval == 0)
            {
                float rewardAvg = 0.0f;

                while (rewardQueue.Count > 0)
                {
                    rewardAvg += rewardQueue.Dequeue();
                }

                rewardAvg /= PrintInterval;

                print("Reward: " + (epi + 1) + " " + rewardAvg + " " + isTraining);
                //print("Loss: " + agent.GetTrainingLoss());
            }

            //print("Episode: " + epi);
            yield return null;
        }

        m_currentCoroutine = null;
        m_coroutineCount++;
    }

    private DeviceDescriptor m_device;
    private Coroutine m_currentCoroutine;
    private int m_coroutineCount = 0;

    private Environment m_environment;
    private Agent m_agent;

    private List<float> m_allRewards;
}


