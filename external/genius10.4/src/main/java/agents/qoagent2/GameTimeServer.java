package agents.qoagent2;

//file name: GameTimeServer.java
import javax.swing.*;
import java.io.*;
import java.net.*;

/*****************************************************************
 * Class name: GameTimeServer
 * Goal: Creating the stop-watch using threads.
 * Input: None.
 * Output: None.
 ****************************************************************/
class GameTimeServer extends JPanel implements Runnable
{
	private static final long serialVersionUID = -7439433284667379204L;
	//DT:	private ServerThread m_st = null; //Allows access to the agent's thread at the server
//DT:	private MultiServer m_server; //Allows access to the server
	private Agent m_agent; //Allows access to the agent running the stop watch
	private int m_nMaxTurn; //Number of turns in the negotiation
	private boolean m_bIsTurn; //True value in this variable means that the 
	//stop watch is used for end-turn. False value means it is used for 
	//end-negotiation.
	private long m_nTime; //the timer
	private JLabel m_lTime; //label for displaying the stop-watch
	private boolean m_bRun; //should the clock continue to run?
	private boolean m_bCountUp; //specifies whether the clock counts down or up.
	private String m_strStartTime; //the text in m_lTime label
	private int m_nStartSeconds; //seconds value
	private int m_nStartMinutes; //minutes value
	private int m_nStartHours; //hours value
		
	
	/*****************************************************************
	* Method name: GameTimeServer()
	* Goal: Constructor.
	* Description: Initialize the class variables. The timer itself is 
	* set by the values of the variables: hours, minutes and seconds.
	* Input: A boolean which specifies whether the stop watch counts up or not,
	* three integers (hours, minutes and seconds), an Agent (the one running 
	* the stop watch), another boolean which specifies whether the stop watch is 
	* of type end-turn or end-negotiation, another integer (number of turns in 
	* the negotiation), a MultiServer object and a ServerThread object.
	* Output: None.
	****************************************************************/
	public GameTimeServer(boolean bCountUp,int nHours,int nMinutes, int nSeconds,Agent agent,boolean bTurnOrNeg,int nMaxTurn /* DT: ,MultiServer server,ServerThread st */)
	{
//DT:		m_st=st;
//DT:		m_server=server;
		m_nMaxTurn=nMaxTurn;
		m_agent=agent;
		m_bIsTurn=bTurnOrNeg;
		m_bCountUp = bCountUp;
		m_bRun = false;

		if (!bCountUp)
		{
			m_nStartMinutes = nMinutes;
			m_nStartSeconds = nSeconds;
			m_nStartHours = nHours;

			String sec, min, hour;

			m_nTime = nHours*3600+ nMinutes*60 + nSeconds;

			// initializing the seconds and minutes variables
			if ((m_nTime % 60) <10)
				sec = "0"+ m_nTime % 60;
			else
				sec = m_nTime % 60 + "";

			if (( (m_nTime/60) % 60) <10)
				min = "0"+ (m_nTime/60) % 60;
			else
				min = (m_nTime/60) % 60 + "";

			if ((m_nTime / 3600) <10)
				hour = "0"+ m_nTime / 3600;
			else
				hour = m_nTime / 3600 + "";

			m_strStartTime = hour + ":" + min + ":" + sec;
		}
		else
		{
			m_nStartSeconds = 0;
			m_nStartMinutes = 0;
			m_nStartHours = 0;
			m_strStartTime = "00:00";
			m_nTime = 0;
		}

		m_lTime = new JLabel(m_strStartTime);
		add(m_lTime);
	}

/*****************************************************************
 * Method name: GameTimeServer()
 * Goal: default Constructor.
 * Description: Initialize the class variables.
 * Input: None.
 * Output: None.
 ****************************************************************/
	public GameTimeServer()
	{
		m_bCountUp = true;
		m_bRun = false;
		m_nTime = 0;
		m_strStartTime = "00:00:00";

		m_lTime = new JLabel(m_strStartTime);
		add(m_lTime);
	}

/*****************************************************************
 * Method name: run()
 * Goal: run the stop-watch.
 * Description: Increasing/decreasing the time every second while the 
 * game is in progress.
 * Input: None.
 * Output: None.
 ****************************************************************/
	public void run()
	{
		while(m_bRun)
		{
			String sec, min, hour;

			try{
				Thread.sleep(1000);
				if (m_bCountUp)
					m_nTime++; // increasing the time each 1 sec.
				else
					m_nTime--; // decreasing the time each 1 sec.
			} catch(Exception e)
			{
				System.out.println("ERROR----" + e.getMessage() + " [GameTimeServer::run(135)]");
				System.err.println("ERROR----" + e.getMessage() + " [GameTimeServer::run(135)]");
			}

			// initializing the seconds and minutes variables
			if ((m_nTime % 60) <10)
				sec = "0"+ m_nTime % 60;
			else
				sec = m_nTime % 60 + "";

			if (( (m_nTime/60) % 60) <10)
				min = "0"+ (m_nTime/60) % 60;
			else
				min = (m_nTime/60) % 60 + "";

			if ((m_nTime / 3600) <10)
				hour = "0"+ m_nTime / 3600;
			else
				hour = m_nTime / 3600 + "";

			String t = hour + ":" + min + ":" + sec;
			m_lTime.setText(t);

			if (m_nTime == 0)
				stopRunning(); //stop the timer when it reaches 0.
		}
	}

/*****************************************************************
 * Method name: stopRunning()
 * Goal: Stop the stop-watch. If the stop watch is of type end-turn - 
 * we send an endTurn message to the client program. If the stop watch 
 * is of type end-negotiation - we send an endNegotiation message to the 
 * client program, createFrom the log file and remove the client from the
 * clients vector at the server.
 * Input: None.
 * Output: None.
 ****************************************************************/
	public void stopRunning()
	{
		m_bRun = false;
		
		int nCurrentTurn = m_agent.getCurrentTurn();
		
		try
		{
			Socket socket = m_agent.getSocket();
			
			PrintWriter out=new PrintWriter(socket.getOutputStream(),true);
			if(m_bIsTurn)
			{
				out.println("type endTurn");
				System.out.println("COMM-------" + "[Agent " + m_agent.getId() + "] " + "->type endTurn " + nCurrentTurn);
				System.err.println("COMM-------" + "[Agent " + m_agent.getId() + "] " + "->type endTurn " + nCurrentTurn);
			}
			else
			{
/* DT:				String msg=m_st.buildEndNegMessageForAgent(m_agent,"time");
												
				/////make log file///////
				m_st.makeLog(m_agent,false);
*/				
				if(m_agent.getSide().equals("Mediator"))
				{
					try{
						Thread.sleep(2000);	
					}
					catch(Exception e){}
				}
/* DT:							
				//m_server.removeAgentAt(m_server.getAgentIndex(m_agent));
				m_server.removeAgentById(m_agent);
				
				if (m_st != null)
					m_st.setShouldStop(true);
*/					
			}
		}
		catch(IOException e)
		{
			System.out.println("ERROR----" + "[Agent " + m_agent.getId() + "] " + "Can't write to socket: " + e.getMessage() + " [GameTimeServer::stopRunning(208)]");
			System.err.println("ERROR----" + "[Agent " + m_agent.getId() + "] " + "Can't write to socket: " + e.getMessage() + " [GameTimeServer::stopRunning(208)]");
			//System.exit(1);
		}

		if((m_bIsTurn)&&(nCurrentTurn<m_nMaxTurn-1))
		{
			m_agent.incrementCurrentTurn();
			
			newGame(); //restart timer since the negotiation is not over yet.
			m_bRun = true;
		}
		else if((m_bIsTurn)&&(nCurrentTurn==m_nMaxTurn-1))
		{
			m_agent.incrementCurrentTurn();
		}
	}

/*****************************************************************
 * Method name: newGame()
 * Goal: Start a new stop-watch.
 * Input: None.
 * Output: None.
 ****************************************************************/
	public void newGame()
	{
		m_lTime.setText(m_strStartTime);
		m_bRun = true;
		m_nTime = m_nStartHours*3600 + m_nStartMinutes * 60 + m_nStartSeconds;
	}

/*****************************************************************
 * Method name: getTimeLabel()
 * Goal: Return the label with the time.
 * Input: None.
 * Output: Label lTime.
 ****************************************************************/
	public JLabel getTimeLabel()
	{
		return m_lTime;
	}

	/*****************************************************************
	* Method name: setRunMethod()
	* Goal: Setting the timer to count up or down.
	* Input: A boolean (True=up, false=down).
	* Output: None.
	****************************************************************/
	public void setRunMethod(boolean bCountUp)
	{
		m_bCountUp = bCountUp;
	}
	
	/*****************************************************************
	* Method name: setRun()
	* Goal: Setting the timer to run or stop.
	* Input: A boolean (True=run, false=stop).
	* Output: None.
	****************************************************************/
	public void setRun(boolean bRun)
	{
		m_bRun = bRun;
	}

} // end class - GameTime
