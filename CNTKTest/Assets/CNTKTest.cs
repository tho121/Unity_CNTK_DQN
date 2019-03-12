using CNTK;
using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CNTKTest : MonoBehaviour {

    const int EpisodeCount = 100;
    const int MaxSteps = 20;
    const float MinEpsillon = 0.05f;

	// Use this for initialization
	void Start () {
        var device = DeviceDescriptor.UseDefaultDevice();
        print($"Hello from CNTK for {device.Type} only!");

        var env = new Environment();

        var stateSize = env.GetStateSize();
        var actionSize = env.GetActionSize();

        var agent = new Agent(stateSize, actionSize, 64);
        var epsillon = 1.0f;
        
        for(int episodeCount = 0; episodeCount < EpisodeCount; ++episodeCount)
        {
            env.Reset();

            epsillon = Mathf.Max(MinEpsillon, 1.0f - (episodeCount / (EpisodeCount / 2)));

            float episodeReward = 0.0f;
            var currentState = Array.ConvertAll<int, float>(env.GetCurrentState(), x => Convert.ToSingle(x));

            for (int timeStep = 0; timeStep < MaxSteps; ++timeStep)
            {
                var action = agent.Act(currentState, epsillon, actionSize, device);
                float reward = 0.0f;

                bool isFinished = env.Act((Environment.Actions)action, out reward);

                episodeReward += reward;

                var nextState = Array.ConvertAll<int, float>(env.GetCurrentState(), x => Convert.ToSingle(x));

                agent.Observe(currentState, (float)action, reward, nextState, isFinished ? 1.0f : 0.0f);
                agent.Train(64, 0.99f, device);

                if(isFinished)
                {
                    break;
                }

                currentState = nextState;
            }

            print("Reward: " + episodeCount + " " + episodeReward);
        }


    }
    
}


