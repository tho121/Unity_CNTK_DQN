using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using CNTK;
using System;

public class VPGTest : MonoBehaviour {

    const int MaxEpisodes = 1000;
    const int LayerSize = 128;
    const int MaxSteps = 250;
    const float Gamma = 0.95f;

    public bool isTraining = true;

    public VPGEnvironment env;

    private void Awake()
    {
        env.selfReseting = false;

        m_device = DeviceDescriptor.UseDefaultDevice();
        print($"Hello from CNTK for {m_device.Type} only!");

        m_actor = new VPGActor(env.GetStateSize(), env.GetActionSize(), LayerSize);
        m_critic = new VPGCritic(env.GetStateSize(), env.GetActionSize(), LayerSize);

        m_memory = new DQN.Memory(env.GetStateSize());
    }

    // Use this for initialization
    void Start () {

        var currentState = env.GetState();

        //set initial rotation
        m_prevActions = new float[env.GetActionSize()];
        m_prevState = new float[env.GetStateSize()];

        Array.Copy(currentState, m_prevState, currentState.Length);
    }

    //read state, then send action, physics update, read resulting state, create experience
    private void FixedUpdate()
    {
        if(m_currentEpisode < MaxEpisodes)
        {
            var currentState = env.GetState();

            var reward = 1.0f * Time.fixedDeltaTime;

            m_currentRewards += reward;

            //s,a,r,s',d => 5,2,1,5,1 => 15
            //int expSize = m_prevState.Length + m_prevActions.Length + 1 + currentState.Length + 1;

            //float[] exp = new float[expSize];
            //Array.Copy(m_prevState, exp, m_prevState.Length);
            //Array.Copy(m_prevActions, 0, exp, m_prevState.Length, m_prevActions.Length);
            //exp[m_prevState.Length + m_prevActions.Length] = reward;
            //Array.Copy(currentState, 0, exp, m_prevState.Length + m_prevActions.Length + 1, currentState.Length);
            //exp[14] = env.IsDone() ? 1.0f : 0.0f;

            //m_memory.Add(exp);

            var td_target = m_critic.Predict(currentState, m_device);
            var td_error = m_critic.Predict(m_prevState, m_device);

            for (int i = 0; i < td_target.Length; ++i)
            {
                td_target[i] *= Gamma;
                td_target[i] += reward;

                td_error[i] *= -1;
                td_error[i] += td_target[i];
            }

            m_critic.Train(m_prevState, td_target, m_device);
            m_actor.Train(m_prevState, td_error, m_prevActions, m_device);

            if (!env.IsDone())
            {
                float[] actions = m_actor.Act(currentState, m_device);

                m_prevActions = actions;

                float xRot = actions[0];
                float zRot = actions[1];

                env.Act(xRot, xRot);
            }
            else
            {
                env.Reset();
                m_currentEpisode++;
                m_currentRewards = 0.0f;
            }

            //todo: make sure this works with both conditions
            m_prevState = currentState;
        }
    }

    private DeviceDescriptor m_device;

    private int m_currentEpisode = 0;
    private float m_currentRewards = 0.0f;
    private float[] m_prevActions;
    private float[] m_prevState;

    private List<float[]> m_trajectory = new List<float[]>(MaxSteps);

    private VPGActor m_actor;
    private VPGCritic m_critic;
    private DQN.Memory m_memory;
}
