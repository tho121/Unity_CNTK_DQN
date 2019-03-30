using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using CNTK;

namespace VPG
{
    public class Agent
    {
        public Agent(int stateSize, int actionSize, int layerSize)
        {
            m_stateSize = stateSize;
            m_actionSize = actionSize;

            //m_model = Model.CreateNetwork(stateSize, actionSize, layerSize, out m_inputVariable);

            
        }

        //public float[] Act(float[] state)
        //{

        //}
        private int m_stateSize;
        private int m_actionSize;

        private Function m_model;
        private Variable m_inputVariable;
    }


}


