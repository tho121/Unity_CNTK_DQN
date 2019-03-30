using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class VPGEnvironment : MonoBehaviour {

    public Transform platform;
    public Transform ball;

    public float MaxAngle = 45.0f;

    public bool selfReseting = false;

    //public Vector3 testRot;

	// Use this for initialization
	void Start () {

        m_ballStartPos = ball.position;
        m_platformRB = platform.GetComponent<Rigidbody>();

        Reset();
    }

    // Update is called once per frame
    void Update () {
		
	}

    private void FixedUpdate()
    {
        if(selfReseting && IsDone())
        {
            Reset();
        }

        //Act(testRot.x, testRot.z);
    }

    public void Reset()
    {
        platform.Rotate(new Vector3(
            Random.Range(-MaxAngle, MaxAngle),
            0.0f,
            Random.Range(-MaxAngle, MaxAngle)),
            Space.World);

        ball.position = m_ballStartPos;
    }

    public bool IsDone()
    {
        return ball.transform.position.y < -0.5f;
    }

    public float[] GetState()
    {
        m_state[0] = ball.position.x;
        m_state[1] = ball.position.y;
        m_state[2] = ball.position.z;

        m_state[3] = platform.rotation.x;
        m_state[4] = platform.rotation.z;

        return m_state;
    }

    public int GetStateSize()
    {
        return m_state.Length;
    }

    public int GetActionSize()
    {
        return 2;   //x rotation and z rotation
    }

    //values from -1 to 1
    public void Act(float rotX, float rotZ)
    {
        m_platformRB.MoveRotation(Quaternion.Euler(rotX * MaxAngle, 0.0f, rotZ * MaxAngle));
    }

    private Vector3 m_ballStartPos;
    private Rigidbody m_platformRB;
    private float[] m_state = new float[5];
}
