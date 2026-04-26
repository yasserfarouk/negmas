package agents.qoagent2;

import java.util.StringTokenizer;

/*
 * Created on 30/05/2004
 *
 */

/**
 * @author raz
 * @version 1.0
 * @see QOAgent
 * @see QCommunication
 */
public class QMessages 
{
	// constants for probability receiveMessage
	public final static int MESSAGE_RECEIVED = 0;
	public final static int MESSAGE_REJECTED = 1;
	
	public final static int REGISTER = 0;
	public final static int THREAT = 1;
	public final static int COMMENT = 2;
	public final static int OFFER = 3;
	public final static int PROMISE = 4;
	public final static int QUERY = 5;
	public final static int ACCEPT = 6;
	public final static int REJECT = 7;
	public final static int OPT_OUT = 8;
	public final static int COUNTER_OFFER = 9;
	
	private QOAgent m_agent;
	
	/**
	 * @param agent - saves the QOAgent in the member variable
	 */
	public QMessages(QOAgent agent)
	{
		m_agent = agent;
	}

	/**
	 * Formats the message in the predefined structure for sending it
	 * later to the server
	 * @param nMessageKind - the message kind. Can be either:
	 * 	REGISTER, THREAT, COMMENT, OFFER, PROMISE, QUERY, ACCEPT, REJECT, OPT_OUT, COUNTER_OFFER.
	 * @param sMsgBody - the message body: additional data for creating the message.
	 * sMsgBody differs for the different message types
	 * @return the formatted message
	 * @see QOAgent
	 * @see QCommunication
	 */
	public String formatMessage(int nMessageKind, String sMsgBody)
	{
		String sFormattedMsg = "";
		
		switch (nMessageKind)
		{
			case REGISTER:
			{
				sFormattedMsg = "type register tag " + m_agent.getMsgId() + " id " + 
					sMsgBody + " side " + m_agent.getAgentSide() + " name " +  m_agent.getAgentName() +
					" supportMediator " + m_agent.getSupportMediator() + 
					" preferenceDetails automatedAgent";
				;
			}
				break;			
			case THREAT:
			{
				sFormattedMsg = "type threat" +
				" source " + m_agent.getAgentId() +
				" target " + m_agent.getOpponentAgentId() + 
				" tag " + m_agent.getMsgId() + 
				" body "+ sMsgBody;
			}
				break;
			case COMMENT:
			{
				sFormattedMsg = "type comment" +
				" source " + m_agent.getAgentId() +
				" target " + m_agent.getOpponentAgentId() + 
				" tag " + m_agent.getMsgId() + 
				" body "+ sMsgBody;
			}
				break;
			case OFFER:
			{	
				sFormattedMsg = "type offer" +
						" source " + m_agent.getAgentId() +
						" target " + m_agent.getOpponentAgentId() + 
						" tag " + m_agent.getMsgId() + 
						" issueSet ";

				sFormattedMsg += sMsgBody;
			}
				break;
			case COUNTER_OFFER:
			{			
				sFormattedMsg = "type counter_offer" +
						" source " + m_agent.getAgentId() +
						" target " + m_agent.getOpponentAgentId() + 
						" tag " + m_agent.getMsgId() + 
						" issueSet ";

				sFormattedMsg += sMsgBody;
			}
				break;
			case PROMISE:
			{
				sFormattedMsg = "type promise" + 
				" source " + m_agent.getAgentId() +
				" target " + m_agent.getOpponentAgentId() + 
				" tag " + m_agent.getMsgId(); 
				
				// build the agent's issue set
				String sAgentPromise = " myIssueSet ";
				
				// NOTE: In our scenario there are no actions
				// that only for one side.
				// We do not use the option of myIssueSet and yourIssueSet
				sAgentPromise += sMsgBody;

				/*
				QPromiseType agentPromise = m_agent.getPromiseList();
 
				String sAttribute, sValue;
				ArrayList agentPromiseList = (ArrayList)agentPromise.agentIssueSet;
				for (int i = 0; i < agentPromiseList.size(); ++i)
				{
					QAttributeValue av = (QAttributeValue)agentPromiseList.get(i);
					sAttribute = av.sAttribute;
					sValue= av.sValue;

					sAgentPromise += sValue + "*" + sAttribute + "*";
				}
				*/
				sFormattedMsg += sAgentPromise + " ";

				// build the opponent's issue set
				String sOpponentPromise = "yourIssueSet ";
				
				/*
				ArrayList opponentPromiseList = (ArrayList)agentPromise.opponentIssueSet;			

				for (int i = 0; i < opponentPromiseList.size(); ++i)
				{
					QAttributeValue av = (QAttributeValue)opponentPromiseList.get(i);
					sAttribute = av.sAttribute;
					sValue= av.sValue;
	
					sOpponentPromise += sValue + "*" + sAttribute + "*";
				}
				*/
				sFormattedMsg += sOpponentPromise;
			}
				break; 
			case QUERY:
			{
				sFormattedMsg = "type query" +
						" source " + m_agent.getAgentId() +
						" target " + m_agent.getOpponentAgentId() + 
						" tag " + m_agent.getMsgId() + 
						" issueSet ";

				sFormattedMsg += sMsgBody;
			}
				break;
			case ACCEPT:
			{
				sFormattedMsg = "type response" + 
				" source " + m_agent.getAgentId() +
				" target " + m_agent.getOpponentAgentId() + 
				" tag " + m_agent.getMsgId() + 
				" answer AGREE" + 
				" message " + sMsgBody + 
				" reason "; // NOTE: No reason is supplied;
				
				// save accepted msg
				String sResponse = sFormattedMsg.substring(sFormattedMsg.indexOf("answer ")+7, sFormattedMsg.indexOf("reason")-1);
				String sMessage = sResponse.substring(sResponse.indexOf("message ") + 8);
				
				//TODO: Do more than just save accepted offer?
					
				// message accepted - save message
				// parse message by its type (offer, promise, query)
				String sSavedMsg = "";
				if (sMessage.startsWith("type query"))
					sSavedMsg = sMessage.substring(sMessage.indexOf("issueSet ") + 9);
				else if (sMessage.startsWith("type counter_offer"))
					sSavedMsg = sMessage.substring(sMessage.indexOf("issueSet ") + 9);
				else if (sMessage.startsWith("type offer"))
					sSavedMsg = sMessage.substring(sMessage.indexOf("issueSet ") + 9);
				else if (sMessage.startsWith("type promise"))
				{
					String sPromise = sMessage.substring(sMessage.indexOf("myIssueSet ") + 11);
					String sMyIssueSet = sPromise.substring(0, sPromise.indexOf("yourIssueSet "));
					String sYourIssueSet = sPromise.substring(sPromise.indexOf("yourIssueSet ") + 13);

					// parse to one agreement
					sSavedMsg = sMyIssueSet + sYourIssueSet;
				}
				
				// only if accepted an offer - save it
				if ( sMessage.startsWith("type counter_offer") || sMessage.startsWith("type offer"))
					m_agent.saveAcceptedMsg(sSavedMsg);
			}
				break;
			case REJECT:
			{
				sFormattedMsg = "type response" + 
				" source " + m_agent.getAgentId() +
				" target " + m_agent.getOpponentAgentId() + 
				" tag " + m_agent.getMsgId() + 
				" answer DISAGREE" + 
				" message " + sMsgBody + 
				" reason "; // NOTE: No reason is supplied
			}
				break;
			case OPT_OUT:
			{
				sFormattedMsg = "type opt-out tag " + m_agent.getMsgId();
			}
				break;
			default:
			{
				System.out.println("[QO]ERROR: Invalid message kind: " + nMessageKind + " [QMessages::formatMessage(199)]");
				System.err.println("[QO]ERROR: Invalid message kind: " + nMessageKind + " [QMessages::formatMessage(199)]");
			}
				break;
		}

		return sFormattedMsg;
	}

	/**
	 * Parses messages from the server.
	 * NOTE: There is no validation that this agent is the
	 * target for the message. Assuming correctness of server routing messages.
	 * @param sServerLine - the server's message
	 * @return the parsed string - relevant only if "nak"
	 * @see QOAgent
	 * @see QCommunication
	 */
	public String parseMessage(String sServerLine)
	{
		String sParsedString = "";
		
		if (sServerLine.startsWith("type comment"))
		{
			String sComment=sServerLine.substring(sServerLine.indexOf(" body ")+6);
					
			// TODO: use comments?
		}
		else if (sServerLine.startsWith("type threat"))
		{
			String sThreat=sServerLine.substring(sServerLine.indexOf(" body ")+6);
					
			// TODO: use threats?
		}
		else if (sServerLine.startsWith("type endTurn"))
		{
			// turn ended
			m_agent.incrementCurrentTurn();
		}
		else if (sServerLine.startsWith("type endNegotiation"))
		{
			// negotiation ended
			m_agent.m_gtStopTurn.setRun(false);
			m_agent.m_gtStopNeg.setRun(false);
			
			String sEndNegDetails = sServerLine.substring(sServerLine.indexOf("whyEnded"));
			
			System.out.println("[QO]Negotiation Ended");
			System.err.println("[QO]Negotiation Ended");
			
			m_agent.endNegotiation();
			
			// NOTE: no need to parse the end reason, agreement
			// 		and score. They are saved in the file.
		}
		else if (sServerLine.startsWith("type response"))
		{
			String sResponse = sServerLine.substring(sServerLine.indexOf("answer ")+7, sServerLine.indexOf("reason")-1);
			String sAnswerType = sResponse.substring(0,sResponse.indexOf(" "));
			String sMessage = sResponse.substring(sResponse.indexOf("message ") + 8);
			
			String sReason = sServerLine.substring(sServerLine.indexOf("reason ") + 7);

			if(sAnswerType.equals("AGREE"))
			{
				//TODO: Do more than just save accepted offer?
				
				// message accepted - save message
				// parse message by its type (offer, promise, query)
				String sSavedMsg = "";
				if (sMessage.startsWith("type query"))
					sSavedMsg = sMessage.substring(sMessage.indexOf("issueSet ") + 9);
				else if (sMessage.startsWith("type counter_offer"))
					sSavedMsg = sMessage.substring(sMessage.indexOf("issueSet ") + 9);
				else if (sMessage.startsWith("type offer"))
					sSavedMsg = sMessage.substring(sMessage.indexOf("issueSet ") + 9);
				else if (sMessage.startsWith("type promise"))
				{
					String sPromise = sMessage.substring(sMessage.indexOf("myIssueSet ") + 11);
					String sMyIssueSet = sPromise.substring(0, sPromise.indexOf("yourIssueSet "));
					String sYourIssueSet = sPromise.substring(sPromise.indexOf("yourIssueSet ") + 13);

					// parse to one agreement
					sSavedMsg = sMyIssueSet + sYourIssueSet;
				}

				// only if accepted an offer - save it
				// TODO: if our agent send queries/promises - need to offer it now [if it's still worthwhile for us]
				 if ( sMessage.startsWith("type counter_offer") || sMessage.startsWith("type offer"))
				 	m_agent.saveAcceptedMsg(sSavedMsg);
			}
			else if(sAnswerType.equals("DISAGREE"))
			{
				// message rejected
				// parse message by its type (offer, promise, query)
				String sSavedMsg = "";
				int nMessageType = -1;
				if (sMessage.startsWith("type query"))
				{
					sSavedMsg = sMessage.substring(sMessage.indexOf("issueSet ") + 9);
					nMessageType = QUERY;
				}
				else if (sMessage.startsWith("type counter_offer"))
				{
					nMessageType = OFFER;
					sSavedMsg = sMessage.substring(sMessage.indexOf("issueSet ") + 9);
				}
				else if (sMessage.startsWith("type offer"))
				{
					nMessageType = OFFER;
					sSavedMsg = sMessage.substring(sMessage.indexOf("issueSet ") + 9);
				}
				else if (sMessage.startsWith("type promise"))
				{
					nMessageType = PROMISE;
					String sPromise = sMessage.substring(sMessage.indexOf("myIssueSet ") + 11);
					String sMyIssueSet = sPromise.substring(0, sPromise.indexOf("yourIssueSet "));
					String sYourIssueSet = sPromise.substring(sPromise.indexOf("yourIssueSet ") + 13);

					// parse to one agreement
					sSavedMsg = sMyIssueSet + sYourIssueSet;
				}
				
				// receiveMessage opponent's probability
				int CurrentAgreementIdx[] = new int[QAgentType.MAX_ISSUES];
				CurrentAgreementIdx = m_agent.getAgreementIndices(sSavedMsg);
				
				// Update probability of opponent based on message
				m_agent.updateOpponentProbability(CurrentAgreementIdx, nMessageType, MESSAGE_REJECTED);
			}
		}
		else if (sServerLine.startsWith("type registered"))
		{
			String sSecsForTurn = sServerLine.substring(sServerLine.indexOf("secForTurn ")+11);
			StringTokenizer st = new StringTokenizer(sSecsForTurn);
					
			long lSecondsForTurn = Long.parseLong(st.nextToken());
			m_agent.setSecondsForTurn(lSecondsForTurn);

			String sMaxTurns=sServerLine.substring(sServerLine.indexOf("maxTurn ")+8);
			st = new StringTokenizer(sMaxTurns);
			m_agent.setMaxTurns(Integer.parseInt(st.nextToken()));

			long lTotalSec=lSecondsForTurn;
			int nHours=(int)lTotalSec/3600;

			lTotalSec -= nHours*3600;
			int nMinutes=(int)lTotalSec/60;

			lTotalSec -= nMinutes*60;

			m_agent.m_gtStopTurn = new QGameTime(false,nHours,nMinutes,(int)lTotalSec,m_agent,true);
			m_agent.m_gtStopTurn.newGame(); // initializing the stop-watch

			new Thread(m_agent.m_gtStopTurn).start();

			lTotalSec=lSecondsForTurn * m_agent.getMaxTurns();
			nHours=(int)lTotalSec/3600;

			lTotalSec -= nHours*3600;
			nMinutes=(int)lTotalSec/60;

			lTotalSec -= nMinutes*60;

			m_agent.m_gtStopNeg = new QGameTime(false,nHours,nMinutes,(int)lTotalSec,m_agent,false);
			m_agent.m_gtStopNeg.newGame(); // initializing the stop-watch

			new Thread(m_agent.m_gtStopNeg).start();

			String sAgentID = sServerLine.substring(sServerLine.indexOf("agentID ")+8);
			
			m_agent.setHasOpponent(true, sAgentID);
			
			m_agent.calculateFirstOffer();
		}
		else if (sServerLine.startsWith("type agentOptOut"))
		{
			m_agent.setHasOpponent(false, null);

			m_agent.m_gtStopTurn.setRun(false);
			m_agent.m_gtStopNeg.setRun(false);
		}
		else if (sServerLine.equals("type log request error"))
		{
			// not relevant for the QOAgent
		}
		else if (sServerLine.startsWith("type log response"))
		{
			// not relevant for the QOAgent
		}
		else if (sServerLine.startsWith("type query"))
		{
			// query received
			
			// get message id
			String sMsgId;
			StringTokenizer st = new StringTokenizer(sServerLine);
			
			boolean bFound = false;
			while (st.hasMoreTokens() && !bFound)
			{
				if (st.nextToken().equals("tag"))
				{
					bFound = true;
					sMsgId = st.nextToken();
				}
			}
	
			String sQuery=sServerLine.substring(sServerLine.indexOf("issueSet ")+9);
			
			int CurrentAgreementIdx[] = new int[QAgentType.MAX_ISSUES];
			CurrentAgreementIdx = m_agent.getAgreementIndices(sQuery);
			
			m_agent.calculateResponse(QUERY, CurrentAgreementIdx, sServerLine);
		}
		else if (sServerLine.startsWith("type counter_offer"))
		{
			// counter_offer received
			
			// get message id
			String sMsgId;
			StringTokenizer st = new StringTokenizer(sServerLine);
			
			boolean bFound = false;
			while (st.hasMoreTokens() && !bFound)
			{
				if (st.nextToken().equals("tag"))
				{
					bFound = true;
					sMsgId = st.nextToken();
				}
			}
	
			String sOffer=sServerLine.substring(sServerLine.indexOf("issueSet ")+9);
			
			int CurrentAgreementIdx[] = new int[QAgentType.MAX_ISSUES];
			CurrentAgreementIdx = m_agent.getAgreementIndices(sOffer);
			m_agent.calculateResponse(COUNTER_OFFER, CurrentAgreementIdx, sServerLine);
		}
		else if (sServerLine.startsWith("type offer"))
		{
			// offer received
			
			// get message id
			String sMsgId;
			StringTokenizer st = new StringTokenizer(sServerLine);
			
			boolean bFound = false;
			while (st.hasMoreTokens() && !bFound)
			{
				if (st.nextToken().equals("tag"))
				{
					bFound = true;
					sMsgId = st.nextToken();
				}
			}
	
			String sOffer=sServerLine.substring(sServerLine.indexOf("issueSet ")+9);

			int CurrentAgreementIdx[] = new int[QAgentType.MAX_ISSUES];
			CurrentAgreementIdx = m_agent.getAgreementIndices(sOffer);
			
			m_agent.calculateResponse(OFFER, CurrentAgreementIdx, sServerLine);
		}
		else if (sServerLine.startsWith("type promise"))
		{
			// promise received
			
			// get message id
			String sMsgId;
			StringTokenizer st = new StringTokenizer(sServerLine);
			
			boolean bFound = false;
			while (st.hasMoreTokens() && !bFound)
			{
				if (st.nextToken().equals("tag"))
				{
					bFound = true;
					sMsgId = st.nextToken();
				}
			}
			
			String sPromise=sServerLine.substring(sServerLine.indexOf("myIssueSet ")+11);
			String sMyIssueSet=sPromise.substring(0,sPromise.indexOf("yourIssueSet "));
			String sYourIssueSet=sPromise.substring(sPromise.indexOf("yourIssueSet ")+13);

			// parse to one agreement
			int CurrentAgreementIdxMine[] = new int[QAgentType.MAX_ISSUES];
			int CurrentAgreementIdxYours[] = new int[QAgentType.MAX_ISSUES];
			CurrentAgreementIdxMine = m_agent.getAgreementIndices(sMyIssueSet);
			CurrentAgreementIdxYours = m_agent.getAgreementIndices(sYourIssueSet);

			// combine indices
			for (int i = 0; i < QAgentType.MAX_ISSUES; ++i)
			{
				if (CurrentAgreementIdxYours[i] != QAgentType.NO_VALUE)
					CurrentAgreementIdxMine[i] = CurrentAgreementIdxYours[i]; 
			}
			
			m_agent.calculateResponse(PROMISE, CurrentAgreementIdxMine, sServerLine);
		}
		else if (sServerLine.equals("nak") || sServerLine.equals("ack"))
		{
			sParsedString = sServerLine;
		}
		else // other unknown message
		{
			System.out.println("[QO]Unknown Message Error: " + sServerLine + " [QMessages::parseMessage(470)]");
			System.err.println("[QO]Unknown Message Error: " + sServerLine + " [QMessages::parseMessage(470)]");			
			
			sParsedString = sServerLine;
		}
		
		return sParsedString;
	}
}
