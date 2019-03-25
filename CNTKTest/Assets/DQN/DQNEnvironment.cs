using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace DQN
{
    public class Environment
    {
        enum GridSlot
        {
            Current = 0,
            Passable = 1,
            Blocked = 2,
            Finish = 3,
        }

        public enum Actions
        {
            MoveLeft = 0,
            MoveRight = 1,
            MoveUp = 2,
            MoveDown = 3,
        }

        int[] grid = {0, 1, 1, 1,
                  1, 2, 1, 2,
                  1, 1, 1, 3};

        const int NumOfCols = 4;
        const int NumOfRows = 3;

        public Environment()
        {
            Reset();
        }

        public void Reset()
        {
            m_currentGrid = (int[])grid.Clone();

            for (int i = 0; i < m_currentGrid.Length; ++i)
            {
                if (m_currentGrid[i] == 0)
                {
                    m_currentPosition = i;
                }
            }
        }

        public int GetCurrentPosition()
        {
            return m_currentPosition;
        }

        public int[] GetCurrentState()
        {
            return (int[])m_currentGrid.Clone();
        }

        //returns isFinished
        public bool Act(Actions action, out float reward)
        {
            bool isFinished = false;
            reward = -0.1f;

            int row = m_currentPosition / NumOfCols;
            int col = m_currentPosition % NumOfCols;

            switch (action)
            {
                case Actions.MoveLeft:
                    {
                        col = Mathf.Clamp(col - 1, 0, NumOfCols - 1);
                    }
                    break;

                case Actions.MoveRight:
                    {
                        col = Mathf.Clamp(col + 1, 0, NumOfCols - 1);
                    }
                    break;

                case Actions.MoveUp:
                    {
                        row = Mathf.Clamp(row - 1, 0, NumOfRows - 1);
                    }
                    break;

                case Actions.MoveDown:
                    {
                        row = Mathf.Clamp(row + 1, 0, NumOfRows - 1);
                    }
                    break;
            }

            var newPos = row * NumOfCols + col;

            for (int i = 0; i < m_currentGrid.Length; ++i)
            {
                if (i == newPos)
                {
                    if (m_currentGrid[i] == (int)GridSlot.Finish)
                    {
                        isFinished = true;
                        reward = 5.0f;
                        return isFinished;
                    }
                    else if (m_currentGrid[i] == (int)GridSlot.Blocked)
                    {
                        //dont move
                        newPos = m_currentPosition;
                    }

                    break;
                }
            }

            m_currentGrid[m_currentPosition] = (int)GridSlot.Passable;
            m_currentGrid[newPos] = (int)GridSlot.Current;

            m_currentPosition = newPos;

            return isFinished;
        }

        public int GetStateSize()
        {
            return grid.Length;
        }

        public int GetActionSize()
        {
            //Enum.GetNames(typeof(Actions)).Length;
            return 4;
        }

        private int[] m_currentGrid;
        private int m_currentPosition;
    }
}