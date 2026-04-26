package agents.qoagent2;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Random;

import genius.core.issue.ValueDiscrete;
import genius.core.utility.AdditiveUtilitySpace;
import genius.core.utility.EvaluatorDiscrete;


/*
 * Created on 11/09/2004
 *
 */

/**
 * @author raz
 * @version 1.0
 *
 * QAgentsCore class: 
 * In charge of handling the different agent's types.
 * In charge for returning the agreements for the desired type.
 * 
 * @see QOAgent
 * @see QAgentType
 */

public class QAgentsCore {
	//++public static final double NORMALIZE_INCREMENTOR = 20;//TODO: Currently not using normalize incrementor
	
	public static final String COMMENT_CHAR_STR = "#";
	public static final String ISSUE_HEADER_STR = "!";
	public static final String ISSUE_SEPARATOR_STR = "*";	
	public static final String VALUES_UTILITY_SEPARATOR_STR = " ";
	public static final String VALUES_NAMES_SEPARATOR_STR = "~";
	public static final String GENERAL_DATA_SEPARATOR_STR = "@";
	public static final String TIME_EFFECT_STR = "Time-Effect";
	public static final String OPT_OUT_STR = "Opt-Out";
	public static final String STATUS_QUO_STR = "Status-Quo";
	public static final int TIME_EFFECT_IND = 0;
	public static final int OPT_OUT_IND = 1;
	public static final int STATUS_QUO_IND = 2;
	public static final int GENERAL_VALUES_NUM = 3;
	
	
	public static final int LONG_TERM_TYPE_IDX = 0;
	public static final int SHORT_TERM_TYPE_IDX = 1;
	public static final int COMPROMISE_TYPE_IDX = 2;
	public static final int AGENT_TYPES_NUM = 3;
	
	// list of all possible england types - each value is FullUtility
	private ArrayList<QAgentType> m_EnglandAgentTypesList; 
	//	list of all possible zimbabwe types - each value is FullUtility
	private ArrayList<QAgentType> m_ZimbabweAgentTypesList;
	
	// list of all possible england types - each value is FullUtility
	// values for the next turn
	private ArrayList<QAgentType> m_EnglandAgentTypesNextTurnList; 
	//	list of all possible zimbabwe types - each value is FullUtility
	// values for the next turn
	private ArrayList<QAgentType> m_ZimbabweAgentTypesNextTurnList;

	
	private QAgentType m_CurrentAgentType;
	private QAgentType m_CurrentAgentNextTurnType;
	
	private int m_nNextTurnOppType;
	
	private String m_sLogFileName;
	private String m_sProbFileName;
	
	// inner class for calculating QO agreement
	private QGenerateAgreement m_GenerateAgreement;
	private boolean m_bEquilibriumAgent = false;

	
	private agents.QOAgent m_Agent;
	
	public class QGenerateAgreement
	{
		class QCombinedAgreement
		{
			public double m_dAgentAgreementValue;
			public double m_dOpponentAgreementValue;
			public String m_sAgreement;
		}
		
		// selected agreement values
		private double m_dQOValue, m_dNextTurnQOValue, m_dAgentSelectedValue, m_dAgentSelectedNextTurnValue, m_dOppSelectedValue, m_dOppSelectedNextTurnValue;
		private double m_dFirstEquilibriumValue[], m_dFirstEquilibriumValueNextTurn[];
		private double m_dSecondEquilibriumValue[], m_dSecondEquilibriumValueNextTurn[];
		private double m_dEquilibriumValue, m_dNextTurnEquilibriumValue, m_dNextTurnCurrentAgentValueForEqOffer;
		private String m_sAgreement, m_sNextTurnAgreement;
		private String m_sFirstEquilibriumAgreement[];
		private String m_sFirstEquilibriumAgreementNextTurn[];
		private String m_sSecondEquilibriumAgreement[];
		private String m_sSecondEquilibriumAgreementNextTurn[];
		private String m_sEquilibriumAgreement, m_sNextTurnEquilibriumAgreement;
	
		private boolean m_bCalcNashAgreement;
		
		private double m_dBelievedThreshold;
		
		public static final double BELIEVED_THRESHOLD_VAR = 0.1; //TODO: Change threshold
		public static final boolean CALC_NASH_AGREEMENT_VAR = false;
		
		public static final int FIRST_EQUILIBRIUM = 1;
		public static final int SECOND_EQUILIBRIUM = 2;
        private static final int OFFER_SET_SIZE = 4;
		
		public QGenerateAgreement()
		{
			m_dQOValue = QAgentType.VERY_SMALL_NUMBER;
			m_dNextTurnQOValue = QAgentType.VERY_SMALL_NUMBER;
			
			m_nNextTurnOppType = QAgentType.NO_TYPE;
			
			m_sAgreement = "";
			m_sNextTurnAgreement = "";
			
			m_dBelievedThreshold = BELIEVED_THRESHOLD_VAR;
			m_bCalcNashAgreement = CALC_NASH_AGREEMENT_VAR;
			
			m_sEquilibriumAgreement = "";
			m_sNextTurnEquilibriumAgreement = "";
			m_dEquilibriumValue = QAgentType.VERY_SMALL_NUMBER;
			m_dNextTurnEquilibriumValue = QAgentType.VERY_SMALL_NUMBER;
			m_dNextTurnCurrentAgentValueForEqOffer = QAgentType.VERY_SMALL_NUMBER;
			
			m_sFirstEquilibriumAgreement = new String[AGENT_TYPES_NUM + 1];
			m_sFirstEquilibriumAgreementNextTurn = new String[AGENT_TYPES_NUM + 1];
			m_sSecondEquilibriumAgreement = new String[AGENT_TYPES_NUM + 1];
			m_sSecondEquilibriumAgreementNextTurn = new String[AGENT_TYPES_NUM + 1];
			
			m_dFirstEquilibriumValue = new double[AGENT_TYPES_NUM];
			m_dFirstEquilibriumValueNextTurn = new double[AGENT_TYPES_NUM];
			m_dSecondEquilibriumValue = new double[AGENT_TYPES_NUM];
			m_dSecondEquilibriumValueNextTurn = new double[AGENT_TYPES_NUM];
			
			for (int i = 0; i < AGENT_TYPES_NUM; ++i)
			{
				m_sFirstEquilibriumAgreement[i] = "";
				m_sFirstEquilibriumAgreementNextTurn[i] = "";
				m_sSecondEquilibriumAgreement[i] = "";
				m_sSecondEquilibriumAgreementNextTurn[i] = "";
				m_dFirstEquilibriumValue[i] = QAgentType.VERY_SMALL_NUMBER;
				m_dFirstEquilibriumValueNextTurn[i] = QAgentType.VERY_SMALL_NUMBER;
				m_dSecondEquilibriumValue[i] = QAgentType.VERY_SMALL_NUMBER;
				m_dSecondEquilibriumValueNextTurn[i] = QAgentType.VERY_SMALL_NUMBER;
			}
		}
		
		// PRE-CONDITION: m_CurrentAgentType should be updated for the current turn
		public void calculateAgreement(QAgentType agentType, int nCurrentTurn, boolean bCalcForNextTurn)
		{
			if (bCalcForNextTurn)
				m_CurrentAgentNextTurnType = agentType;
			else
				m_CurrentAgentType = agentType;
			
			if (m_CurrentAgentType.isTypeOf(QAgentType.ZIMBABWE_TYPE))
				calculateOfferAgainstOpponent("England", nCurrentTurn, bCalcForNextTurn);
			else if (m_CurrentAgentType.isTypeOf(QAgentType.ENGLAND_TYPE))
				calculateOfferAgainstOpponent("Zimbabwe", nCurrentTurn, bCalcForNextTurn);
			else
			{
				System.out.println("[QO]Agent type is unknown [QAgentsCore::calculateAgreement(204)]");
				System.err.println("[QO]Agent type is unknown [QAgentsCore::calculateAgreement(204)]");
			}
		}
		
		public void calculateOfferAgainstOpponent(String sOpponentType, int nCurrentTurn, boolean bCalcForNextTurn)
		{
			try {
				PrintWriter bw = new PrintWriter(new FileWriter(m_sLogFileName, true));
				
			
			QAgentType agentOpponentCompromise = null;
			QAgentType agentOpponentLongTerm = null;
			QAgentType agentOpponentShortTerm = null;
			
			m_dQOValue = QAgentType.VERY_SMALL_NUMBER;
			m_dNextTurnQOValue = QAgentType.VERY_SMALL_NUMBER;
			
			m_nNextTurnOppType = QAgentType.NO_TYPE;

			int nIssuesNum = m_CurrentAgentType.getIssuesNum();

			//06-05-06
			int OpponentShortTermIdx[][] = new int[OFFER_SET_SIZE][nIssuesNum];
			int OpponentLongTermIdx[][] = new int[OFFER_SET_SIZE][nIssuesNum];
			int OpponentCompromiseIdx[][] = new int[OFFER_SET_SIZE][nIssuesNum];
			
			
			double /*06-05-06 dOpponentShortTermQOValue = QAgentType.VERY_SMALL_NUMBER, */dOpponentShortTermAgreementValue = QAgentType.VERY_SMALL_NUMBER, dOpponentShortTermLuceAgreementValue = 0;
			double /*06-05-06 dOpponentLongTermQOValue = QAgentType.VERY_SMALL_NUMBER, */dOpponentLongTermAgreementValue = QAgentType.VERY_SMALL_NUMBER, dOpponentLongTermLuceAgreementValue = 0;
			double /*06-05-06 dOpponentCompromiseQOValue = QAgentType.VERY_SMALL_NUMBER, */dOpponentCompromiseAgreementValue = QAgentType.VERY_SMALL_NUMBER, dOpponentCompromiseLuceAgreementValue = 0;
			double dQOValue = QAgentType.VERY_SMALL_NUMBER;

			//06-05-06
			double dOpponentShortTermQOValue[] = new double[OFFER_SET_SIZE];
			double dOpponentLongTermQOValue[] = new double[OFFER_SET_SIZE];
			double dOpponentCompromiseQOValue[] = new double[OFFER_SET_SIZE];
			
			for (int i = 0; i < dOpponentShortTermQOValue.length; ++i) {
			    dOpponentShortTermQOValue[i] = QAgentType.VERY_SMALL_NUMBER;
			    dOpponentLongTermQOValue[i] = QAgentType.VERY_SMALL_NUMBER;
			    dOpponentCompromiseQOValue[i] = QAgentType.VERY_SMALL_NUMBER;
			}
			
			// nash variables
			double dCurrentNashValue = QAgentType.VERY_SMALL_NUMBER;
			double dOpponentCompromiseNashValue = QAgentType.VERY_SMALL_NUMBER; 
			double dOpponentLongTermNashValue = QAgentType.VERY_SMALL_NUMBER;
			double dOpponentShortTermNashValue = QAgentType.VERY_SMALL_NUMBER;
			
			double dBelievedTypeOpponentShortTerm = 0;
			double dBelievedTypeOpponentLongTerm = 0;
			double dBelievedTypeOpponentCompromise = 0;
			
			boolean bCalcOpponentShortTerm = false;
			boolean bCalcOpponentLongTerm = false;
			boolean bCalcOpponentCompromise = false;
			
			double dCurrentAgentAgreementValue = QAgentType.VERY_SMALL_NUMBER, dCurrentAgentLuceAgreementValue = QAgentType.VERY_SMALL_NUMBER;
			
			int index = -1;//raz 06-05-06
		
			if (sOpponentType.equals("England"))
			{
				if (bCalcForNextTurn)
				{
					agentOpponentCompromise = getEnglandCompromiseNextTurnType();
					agentOpponentLongTerm = getEnglandLongTermNextTurnType();
					agentOpponentShortTerm = getEnglandShortTermNextTurnType();
				}
				else
				{
					agentOpponentCompromise = getEnglandCompromiseType();
					agentOpponentLongTerm = getEnglandLongTermType();
					agentOpponentShortTerm = getEnglandShortTermType();					
				}
			}
			else if (sOpponentType.equals("Zimbabwe"))
			{
				if (bCalcForNextTurn)
				{
					agentOpponentCompromise = getZimbabweCompromiseNextTurnType();
					agentOpponentLongTerm = getZimbabweLongTermNextTurnType();
					agentOpponentShortTerm = getZimbabweShortTermNextTurnType();
				}
				else
				{
					agentOpponentCompromise = getZimbabweCompromiseType();
					agentOpponentLongTerm = getZimbabweLongTermType();
					agentOpponentShortTerm = getZimbabweShortTermType();				
				}
			}
			else
			{					
				System.out.println("[QO]Agent type is unknown [QAgentsCore::calculateOfferAgainstOpponent(291)]");
				System.err.println("[QO]Agent type is unknown [QAgentsCore::calculateOfferAgainstOpponent(291)]");
				return;
			}		
			
			dBelievedTypeOpponentCompromise = agentOpponentCompromise.getTypeProbability();
			dBelievedTypeOpponentLongTerm = agentOpponentLongTerm.getTypeProbability();
			dBelievedTypeOpponentShortTerm = agentOpponentShortTerm.getTypeProbability();
			
			if (dBelievedTypeOpponentCompromise > dBelievedTypeOpponentLongTerm)
			{
				if (dBelievedTypeOpponentCompromise > dBelievedTypeOpponentShortTerm)
				{
					bCalcOpponentCompromise = true;
					
					if (dBelievedTypeOpponentCompromise - dBelievedTypeOpponentShortTerm < m_dBelievedThreshold)
						bCalcOpponentShortTerm = true;
					if (dBelievedTypeOpponentCompromise - dBelievedTypeOpponentLongTerm < m_dBelievedThreshold)
						bCalcOpponentLongTerm = true;
				}
				else
				{
					bCalcOpponentShortTerm = true;
					
					if (dBelievedTypeOpponentShortTerm - dBelievedTypeOpponentCompromise < m_dBelievedThreshold)
						bCalcOpponentCompromise = true;
					if (dBelievedTypeOpponentShortTerm - dBelievedTypeOpponentLongTerm < m_dBelievedThreshold)
						bCalcOpponentLongTerm = true;
				}
			}
			else
			{
				if (dBelievedTypeOpponentLongTerm > dBelievedTypeOpponentShortTerm)
				{
					bCalcOpponentLongTerm = true;
					
					if (dBelievedTypeOpponentLongTerm - dBelievedTypeOpponentShortTerm < m_dBelievedThreshold)
						bCalcOpponentShortTerm = true;
					if (dBelievedTypeOpponentLongTerm - dBelievedTypeOpponentCompromise < m_dBelievedThreshold)
						bCalcOpponentCompromise = true;
				}
				else
				{
					bCalcOpponentShortTerm = true;
					
					if (dBelievedTypeOpponentShortTerm - dBelievedTypeOpponentCompromise < m_dBelievedThreshold)
						bCalcOpponentCompromise = true;
					if (dBelievedTypeOpponentShortTerm - dBelievedTypeOpponentLongTerm < m_dBelievedThreshold)
						bCalcOpponentLongTerm = true;
				}
			}
			
			int CurrentAgreementIdx[] = new int[nIssuesNum];
			int MaxIssueValues[] = new int[nIssuesNum];
			
			for (int i = 0; i < nIssuesNum; ++i)
			{
				CurrentAgreementIdx[i] = 0;
				MaxIssueValues[i] = m_CurrentAgentType.getMaxIssueValue(i);
			}
			
			int nTotalAgreements = m_CurrentAgentType.getTotalAgreements();
			
			for (int i = 0; i < nTotalAgreements; ++i)
			{
				// calculate agreements
				if (bCalcForNextTurn)
				{
					//22-09-05
					// dCurrentAgentAgreementValue = m_CurrentAgentNextTurnType.getAgreementValue(CurrentAgreementIdx, nCurrentTurn);
					dCurrentAgentAgreementValue = m_CurrentAgentNextTurnType.getAgreementRankingProbability(CurrentAgreementIdx);
					double agreementUtilityValue = m_CurrentAgentNextTurnType.getAgreementValue(CurrentAgreementIdx, nCurrentTurn);
					dCurrentAgentLuceAgreementValue = m_CurrentAgentNextTurnType.getAgreementLuceValue(agreementUtilityValue);
					
					//25-09-05
					// take into account only agreements that worth more than the status quo
					if (agreementUtilityValue <= m_CurrentAgentNextTurnType.getSQValue()) {
//					  receiveMessage issue values indices
						boolean bFinishUpdate = false;
						for (int k = nIssuesNum-1; k >= 0 && !bFinishUpdate; --k)
						{
							if (CurrentAgreementIdx[k]+1 >= MaxIssueValues[k])
							{
								CurrentAgreementIdx[k] = 0;
							}
							else
							{
								CurrentAgreementIdx[k]++;
								bFinishUpdate = true;
							}									
						}					    
					    
					    continue;
						
					}
					
					//06-05-06
					// take into account only agreements that are less than the X best agreements
					// (as time passes, the chances of them being accepted are lower)
					/*if (agreementUtilityValue > m_CurrentAgentNextTurnType.getMaxValue()) {
//						  receiveMessage issue values indices
							boolean bFinishUpdate = false;
							for (int k = nIssuesNum-1; k >= 0 && !bFinishUpdate; --k)
							{
								if (CurrentAgreementIdx[k]+1 >= MaxIssueValues[k])
								{
									CurrentAgreementIdx[k] = 0;
								}
								else
								{
									CurrentAgreementIdx[k]++;
									bFinishUpdate = true;
								}									
							}					    
						    
						    continue;
							
						}
						*/					
				}
				else
				{
					//22-09-05
					//dCurrentAgentAgreementValue = m_CurrentAgentType.getAgreementValue(CurrentAgreementIdx, nCurrentTurn);
					dCurrentAgentAgreementValue = m_CurrentAgentType.getAgreementRankingProbability(CurrentAgreementIdx);
					double agreementUtilityValue = m_CurrentAgentType.getAgreementValue(CurrentAgreementIdx, nCurrentTurn);
					dCurrentAgentLuceAgreementValue = m_CurrentAgentType.getAgreementLuceValue(agreementUtilityValue);
					
					//25-09-05
					if (agreementUtilityValue <= m_CurrentAgentType.getSQValue()) {
					    
//					  receiveMessage issue values indices
						boolean bFinishUpdate = false;
						for (int k = nIssuesNum-1; k >= 0 && !bFinishUpdate; --k)
						{
							if (CurrentAgreementIdx[k]+1 >= MaxIssueValues[k])
							{
								CurrentAgreementIdx[k] = 0;
							}
							else
							{
								CurrentAgreementIdx[k]++;
								bFinishUpdate = true;
							}									
						}
						continue;
					}
					//06-05-06
					// take into account only agreements that are less than the X best agreements
					// (as time passes, the chances of them being accepted are lower)
					/*if (agreementUtilityValue > m_CurrentAgentType.getMaxValue()) {
//						  receiveMessage issue values indices
							boolean bFinishUpdate = false;
							for (int k = nIssuesNum-1; k >= 0 && !bFinishUpdate; --k)
							{
								if (CurrentAgreementIdx[k]+1 >= MaxIssueValues[k])
								{
									CurrentAgreementIdx[k] = 0;
								}
								else
								{
									CurrentAgreementIdx[k]++;
									bFinishUpdate = true;
								}									
							}					    
						    
						    continue;
							
						}	
						*/									
				}

				bw.println("----------------------------------------");
				bw.println("Agreement: " + m_CurrentAgentType.getAgreementStr(CurrentAgreementIdx) + "(turn: " + nCurrentTurn + ")");
				bw.println("Agreement Value: " + dCurrentAgentAgreementValue);
				bw.println("Agreement Luce: " + dCurrentAgentLuceAgreementValue);
				
				if (bCalcOpponentCompromise)
				{
					//22-09-05
					//dOpponentCompromiseAgreementValue = agentOpponentCompromise.getAgreementValue(CurrentAgreementIdx, nCurrentTurn);
					dOpponentCompromiseAgreementValue = agentOpponentCompromise.getAgreementRankingProbability(CurrentAgreementIdx);
					double agreementUtilityValue = agentOpponentCompromise.getAgreementValue(CurrentAgreementIdx, nCurrentTurn);
					dOpponentCompromiseLuceAgreementValue = agentOpponentCompromise.getAgreementLuceValue(agreementUtilityValue);
				
					//25-09-05
					if (agreementUtilityValue <= agentOpponentCompromise.getSQValue()) {
//					  receiveMessage issue values indices
						boolean bFinishUpdate = false;
						for (int k = nIssuesNum-1; k >= 0 && !bFinishUpdate; --k)
						{
							if (CurrentAgreementIdx[k]+1 >= MaxIssueValues[k])
							{
								CurrentAgreementIdx[k] = 0;
							}
							else
							{
								CurrentAgreementIdx[k]++;
								bFinishUpdate = true;
							}									
						}
					    
					    continue; // don't compute this agreement
						
					}
					
					bw.println("CompOpponent Value: " + dOpponentCompromiseAgreementValue);
					bw.println("CompOpponent Luce: " + dOpponentCompromiseLuceAgreementValue);
					
					dQOValue = Math.min(
						dCurrentAgentAgreementValue * dCurrentAgentLuceAgreementValue, //@@,
					dOpponentCompromiseAgreementValue * (dOpponentCompromiseLuceAgreementValue + dCurrentAgentLuceAgreementValue)
					);

					bw.println("CompQO Value: " + dQOValue);
					
					index = 0;
					if (dQOValue > dOpponentCompromiseQOValue[0])
					{
					    if (dQOValue > dOpponentCompromiseQOValue[1]) {
					        index = 1;
					        if (dQOValue > dOpponentCompromiseQOValue[2]) {
					            index = 2;
					            if (dQOValue > dOpponentCompromiseQOValue[3]) {
					                index = 3;
					            }
					        }
					    }

					    // 06-05-06
					    // save value
					    if (index == 3) {
					        dOpponentCompromiseQOValue[0] = dOpponentCompromiseQOValue[1];
					        dOpponentCompromiseQOValue[1] = dOpponentCompromiseQOValue[2];
					        dOpponentCompromiseQOValue[2] = dOpponentCompromiseQOValue[3];
					        dOpponentCompromiseQOValue[3] = dQOValue;
					        
					        for (int j = 0; j < nIssuesNum; ++j) {
					            OpponentCompromiseIdx[0][j] = OpponentCompromiseIdx[1][j];
					            OpponentCompromiseIdx[1][j] = OpponentCompromiseIdx[2][j];
								OpponentCompromiseIdx[2][j] = OpponentCompromiseIdx[3][j];
								OpponentCompromiseIdx[3][j] = CurrentAgreementIdx[j];
					        }
					    }
					    else if (index == 2) {
					        dOpponentCompromiseQOValue[0] = dOpponentCompromiseQOValue[1];
					        dOpponentCompromiseQOValue[1] = dOpponentCompromiseQOValue[2];
					        dOpponentCompromiseQOValue[2] = dQOValue;	
					        
					        for (int j = 0; j < nIssuesNum; ++j) {
					            OpponentCompromiseIdx[0][j] = OpponentCompromiseIdx[1][j];
					            OpponentCompromiseIdx[1][j] = OpponentCompromiseIdx[2][j];
					            OpponentCompromiseIdx[2][j] = CurrentAgreementIdx[j];
					        }					        
					    }
					    else if (index == 1) {
					        dOpponentCompromiseQOValue[0] = dOpponentCompromiseQOValue[1];
					        dOpponentCompromiseQOValue[1] = dQOValue;
					        
					        for (int j = 0; j < nIssuesNum; ++j) {
					            OpponentCompromiseIdx[0][j] = OpponentCompromiseIdx[1][j];
					            OpponentCompromiseIdx[1][j] = CurrentAgreementIdx[j];
					        }					        
					    }
					    else { // index == 0
					        dOpponentCompromiseQOValue[0] = dQOValue;
					        
					        for (int j = 0; j < nIssuesNum; ++j) {
					            OpponentCompromiseIdx[0][j] = CurrentAgreementIdx[j];
					        }					        
					    }
					    
						bw.println("------------SAVED VALUE--------------");
						// save value
						//06-05-06 dOpponentCompromiseQOValue = dQOValue;
						
						//06-05-06
						// save agreement
						//for (int j = 0; j < nIssuesNum; ++j)
							//OpponentCompromiseIdx[j] = CurrentAgreementIdx[j];
					}
					if (m_bCalcNashAgreement)
					{
						dCurrentNashValue = dOpponentCompromiseAgreementValue * dCurrentAgentAgreementValue;
								
						if (dCurrentNashValue > dOpponentCompromiseNashValue)
						{
							dOpponentCompromiseNashValue = dCurrentNashValue;
						}
					}
				}
				if (bCalcOpponentLongTerm)
				{
					//22-09-05
					//dOpponentLongTermAgreementValue = agentOpponentLongTerm.getAgreementValue(CurrentAgreementIdx, nCurrentTurn); 
					dOpponentLongTermAgreementValue = agentOpponentLongTerm.getAgreementRankingProbability(CurrentAgreementIdx);
					double agreementUtilityValue = agentOpponentLongTerm.getAgreementValue(CurrentAgreementIdx, nCurrentTurn);
					dOpponentLongTermLuceAgreementValue = agentOpponentLongTerm.getAgreementLuceValue(agreementUtilityValue);

					//25-09-05
					if (agreementUtilityValue <= agentOpponentLongTerm.getSQValue()) {
//					  receiveMessage issue values indices
						boolean bFinishUpdate = false;
						for (int k = nIssuesNum-1; k >= 0 && !bFinishUpdate; --k)
						{
							if (CurrentAgreementIdx[k]+1 >= MaxIssueValues[k])
							{
								CurrentAgreementIdx[k] = 0;
							}
							else
							{
								CurrentAgreementIdx[k]++;
								bFinishUpdate = true;
							}									
						}
					    
					    continue; // don't compute this agreement
					}
					
					bw.println("LongOpponent Value: " + dOpponentLongTermAgreementValue);
					bw.println("LongOpponent Luce: " + dOpponentLongTermLuceAgreementValue);

					dQOValue = Math.min(
						dCurrentAgentAgreementValue * dCurrentAgentLuceAgreementValue, //@@,
					dOpponentLongTermAgreementValue * (dOpponentLongTermLuceAgreementValue + dCurrentAgentLuceAgreementValue)
					);

					bw.println("LongQO Value: " + dQOValue);
					
					index = 0;
					if (dQOValue > dOpponentLongTermQOValue[0])
					{
					    if (dQOValue > dOpponentLongTermQOValue[1]) {
					        index = 1;
					        if (dQOValue > dOpponentLongTermQOValue[2]) {
					            index = 2;
					            if (dQOValue > dOpponentLongTermQOValue[3]) {
					                index = 3;
					            }
					        }
					    }

					    // 06-05-06
					    // save value
					    if (index == 3) {
					        dOpponentLongTermQOValue[0] = dOpponentLongTermQOValue[1];
					        dOpponentLongTermQOValue[1] = dOpponentLongTermQOValue[2];
					        dOpponentLongTermQOValue[2] = dOpponentLongTermQOValue[3];
					        dOpponentLongTermQOValue[3] = dQOValue;
					        
					        for (int j = 0; j < nIssuesNum; ++j) {
					            OpponentLongTermIdx[0][j] = OpponentLongTermIdx[1][j];
								OpponentLongTermIdx[1][j] = OpponentLongTermIdx[2][j];
								OpponentLongTermIdx[2][j] = OpponentLongTermIdx[3][j];
								OpponentLongTermIdx[3][j] = CurrentAgreementIdx[j];
					        }
					    }
					    else if (index == 2) {
					        dOpponentLongTermQOValue[0] = dOpponentLongTermQOValue[1];
					        dOpponentLongTermQOValue[1] = dOpponentLongTermQOValue[2];
					        dOpponentLongTermQOValue[2] = dQOValue;	
					        
					        for (int j = 0; j < nIssuesNum; ++j) {
					            OpponentLongTermIdx[0][j] = OpponentLongTermIdx[1][j];
					            OpponentLongTermIdx[1][j] = OpponentLongTermIdx[2][j];
					            OpponentLongTermIdx[2][j] = CurrentAgreementIdx[j];
					        }					        
					    }
					    else if (index == 1) {
					        dOpponentLongTermQOValue[0] = dOpponentLongTermQOValue[1];
					        dOpponentLongTermQOValue[1] = dQOValue;
					        
					        for (int j = 0; j < nIssuesNum; ++j) {
					            OpponentLongTermIdx[0][j] = OpponentLongTermIdx[1][j];
								OpponentLongTermIdx[1][j] = CurrentAgreementIdx[j];
					        }					        
					    }
					    else { // index == 0
					        dOpponentLongTermQOValue[0] = dQOValue;
					        
					        for (int j = 0; j < nIssuesNum; ++j) {
					            OpponentLongTermIdx[0][j] = CurrentAgreementIdx[j];
					        }					        
					    }
					    
						bw.println("------------SAVED VALUE--------------");
						// save value
						//06-05-06 dOpponentLongTermQOValue = dQOValue;
						
						//06-05-06
						// save agreement
						//for (int j = 0; j < nIssuesNum; ++j)
							//OpponentLongTermIdx[j] = CurrentAgreementIdx[j];
					}
					if (m_bCalcNashAgreement)
					{
						dCurrentNashValue = dOpponentLongTermAgreementValue * dCurrentAgentAgreementValue;
								
						if (dCurrentNashValue > dOpponentLongTermNashValue)
						{
							dOpponentLongTermNashValue = dCurrentNashValue;
						}
					}
				}
				if (bCalcOpponentShortTerm)
				{
					//22-09-05
					//dOpponentShortTermAgreementValue = agentOpponentShortTerm.getAgreementValue(CurrentAgreementIdx, nCurrentTurn); 
					dOpponentShortTermAgreementValue = agentOpponentShortTerm.getAgreementRankingProbability(CurrentAgreementIdx);
					double agreementUtilityValue = agentOpponentShortTerm.getAgreementValue(CurrentAgreementIdx, nCurrentTurn);
					dOpponentShortTermLuceAgreementValue = agentOpponentShortTerm.getAgreementLuceValue(agreementUtilityValue);
					
					//25-09-05
					if (agreementUtilityValue <= agentOpponentShortTerm.getSQValue()) {
//					  receiveMessage issue values indices
						boolean bFinishUpdate = false;
						for (int k = nIssuesNum-1; k >= 0 && !bFinishUpdate; --k)
						{
							if (CurrentAgreementIdx[k]+1 >= MaxIssueValues[k])
							{
								CurrentAgreementIdx[k] = 0;
							}
							else
							{
								CurrentAgreementIdx[k]++;
								bFinishUpdate = true;
							}									
						}
						
					    continue; // don't compute this agreement
					}
								
					bw.println("ShortOpponent Value: " + dOpponentShortTermAgreementValue);
					bw.println("ShortOpponent Luce: " + dOpponentShortTermLuceAgreementValue);

					dQOValue = Math.min(
						dCurrentAgentAgreementValue * dCurrentAgentLuceAgreementValue, //@@,
					dOpponentShortTermAgreementValue * (dOpponentShortTermLuceAgreementValue + dCurrentAgentLuceAgreementValue)
					);
							
					bw.println("ShortQO Value: " + dQOValue);
					
					index = 0;
					if (dQOValue > dOpponentShortTermQOValue[0])
					{
					    if (dQOValue > dOpponentShortTermQOValue[1]) {
					        index = 1;
					        if (dQOValue > dOpponentShortTermQOValue[2]) {
					            index = 2;
					            if (dQOValue > dOpponentShortTermQOValue[3]) {
					                index = 3;
					            }
					        }
					    }

					    // 06-05-06
					    // save value
					    if (index == 3) {
					        dOpponentShortTermQOValue[0] = dOpponentShortTermQOValue[1];
					        dOpponentShortTermQOValue[1] = dOpponentShortTermQOValue[2];
					        dOpponentShortTermQOValue[2] = dOpponentShortTermQOValue[3];
					        dOpponentShortTermQOValue[3] = dQOValue;
					        
					        for (int j = 0; j < nIssuesNum; ++j) {
								OpponentShortTermIdx[0][j] = OpponentShortTermIdx[1][j];
								OpponentShortTermIdx[1][j] = OpponentShortTermIdx[2][j];
								OpponentShortTermIdx[2][j] = OpponentShortTermIdx[3][j];
								OpponentShortTermIdx[3][j] = CurrentAgreementIdx[j];
					        }
					    }
					    else if (index == 2) {
					        dOpponentShortTermQOValue[0] = dOpponentShortTermQOValue[1];
					        dOpponentShortTermQOValue[1] = dOpponentShortTermQOValue[2];
					        dOpponentShortTermQOValue[2] = dQOValue;	
					        
					        for (int j = 0; j < nIssuesNum; ++j) {
								OpponentShortTermIdx[0][j] = OpponentShortTermIdx[1][j];
								OpponentShortTermIdx[1][j] = OpponentShortTermIdx[2][j];
								OpponentShortTermIdx[2][j] = CurrentAgreementIdx[j];
					        }					        
					    }
					    else if (index == 1) {
					        dOpponentShortTermQOValue[0] = dOpponentShortTermQOValue[1];
					        dOpponentShortTermQOValue[1] = dQOValue;
					        
					        for (int j = 0; j < nIssuesNum; ++j) {
								OpponentShortTermIdx[0][j] = OpponentShortTermIdx[1][j];
								OpponentShortTermIdx[1][j] = CurrentAgreementIdx[j];
					        }					        
					    }
					    else { // index == 0
					        dOpponentShortTermQOValue[0] = dQOValue;
					        
					        for (int j = 0; j < nIssuesNum; ++j) {
								OpponentShortTermIdx[0][j] = CurrentAgreementIdx[j];
					        }					        
					    }
					    
						bw.println("------------SAVED VALUE--------------");
						// save value
						//06-05-06 dOpponentShortTermQOValue = dQOValue;
						
						//06-05-06
						// save agreement
						//for (int j = 0; j < nIssuesNum; ++j)
							//OpponentShortTermIdx[j] = CurrentAgreementIdx[j];
					}
					if (m_bCalcNashAgreement)
					{
						dCurrentNashValue = dOpponentShortTermAgreementValue * dCurrentAgentAgreementValue;
								
						if (dCurrentNashValue > dOpponentShortTermNashValue)
						{
							dOpponentShortTermNashValue = dCurrentNashValue;
						}
					}
				}

				// receiveMessage issue values indices
				boolean bFinishUpdate = false;
				for (int k = nIssuesNum-1; k >= 0 && !bFinishUpdate; --k)
				{
					if (CurrentAgreementIdx[k]+1 >= MaxIssueValues[k])
					{
						CurrentAgreementIdx[k] = 0;
					}
					else
					{
						CurrentAgreementIdx[k]++;
						bFinishUpdate = true;
					}									
				}
			} // end for - going over all possible agreements 
			
			Random rand = new Random();
		    
			// select which offer to propose
			// generate a random number between 0 and OFFER_SET_SIZE
		    int randNum = rand.nextInt(OFFER_SET_SIZE);

			if (bCalcForNextTurn)
			{
				if (bCalcOpponentCompromise)
				{
					if (dOpponentCompromiseQOValue[randNum] * dBelievedTypeOpponentCompromise > m_dNextTurnQOValue)
					{
						m_dNextTurnQOValue = dOpponentCompromiseQOValue[randNum] * dBelievedTypeOpponentCompromise;
						m_dAgentSelectedNextTurnValue = m_CurrentAgentNextTurnType.getAgreementValue(OpponentCompromiseIdx[randNum], nCurrentTurn);
						m_dOppSelectedNextTurnValue = agentOpponentCompromise.getAgreementValue(OpponentCompromiseIdx[randNum], nCurrentTurn);
						m_sNextTurnAgreement = m_CurrentAgentType.getAgreementStr(OpponentCompromiseIdx[randNum]);
						m_nNextTurnOppType = COMPROMISE_TYPE_IDX;
					}
				}
				if (bCalcOpponentLongTerm)
				{
					if (dOpponentLongTermQOValue[randNum] * dBelievedTypeOpponentLongTerm > m_dNextTurnQOValue)
					{
						m_dNextTurnQOValue = dOpponentLongTermQOValue[randNum] * dBelievedTypeOpponentLongTerm;
						m_dAgentSelectedNextTurnValue = m_CurrentAgentNextTurnType.getAgreementValue(OpponentLongTermIdx[randNum], nCurrentTurn);
						m_dOppSelectedNextTurnValue = agentOpponentLongTerm.getAgreementValue(OpponentLongTermIdx[randNum], nCurrentTurn);
						m_sNextTurnAgreement = m_CurrentAgentType.getAgreementStr(OpponentLongTermIdx[randNum]);
						m_nNextTurnOppType = LONG_TERM_TYPE_IDX;
					}
				}
				if (bCalcOpponentShortTerm)
				{
					if (dOpponentShortTermQOValue[randNum] * dBelievedTypeOpponentShortTerm > m_dNextTurnQOValue)
					{
						m_dNextTurnQOValue = dOpponentShortTermQOValue[randNum] * dBelievedTypeOpponentShortTerm;
						m_dAgentSelectedNextTurnValue = m_CurrentAgentNextTurnType.getAgreementValue(OpponentShortTermIdx[randNum], nCurrentTurn);
						m_dOppSelectedNextTurnValue = agentOpponentShortTerm.getAgreementValue(OpponentShortTermIdx[randNum], nCurrentTurn);
						m_sNextTurnAgreement = m_CurrentAgentType.getAgreementStr(OpponentShortTermIdx[randNum]);
						m_nNextTurnOppType = SHORT_TERM_TYPE_IDX;
					}
				}
			} // end if - calculate for next turn
			else // calculate for current turn
			{
				String sSelectedOffer = "";
				String sSelectedOpponent = "";
				if (bCalcOpponentCompromise)
				{
					if (dOpponentCompromiseQOValue[randNum] * dBelievedTypeOpponentCompromise > m_dQOValue)
					{
						bw.println("~~~~~~~~~~SELECTED COMP~~~~~~~~~~");
						m_dQOValue = dOpponentCompromiseQOValue[randNum] * dBelievedTypeOpponentCompromise;
						m_dAgentSelectedValue = m_CurrentAgentType.getAgreementValue(OpponentCompromiseIdx[randNum], nCurrentTurn);
						m_dOppSelectedValue = agentOpponentCompromise.getAgreementValue(OpponentCompromiseIdx[randNum], nCurrentTurn);
						m_sAgreement = m_CurrentAgentType.getAgreementStr(OpponentCompromiseIdx[randNum]);
						bw.println("Agreement: " + m_sAgreement);
						bw.println("QO Value: " + m_dQOValue);
						bw.println("Agent Value: " + m_dAgentSelectedValue);
						bw.println("Opponent Value: " + m_dOppSelectedValue);
						
						sSelectedOpponent = "Compromise";
					}
				}
				if (bCalcOpponentLongTerm)
				{
					if (dOpponentLongTermQOValue[randNum] * dBelievedTypeOpponentLongTerm > m_dQOValue)
					{
						bw.println("~~~~~~~~~~SELECTED LONG~~~~~~~~~~");
						m_dQOValue = dOpponentLongTermQOValue[randNum] * dBelievedTypeOpponentLongTerm;
						m_dAgentSelectedValue = m_CurrentAgentType.getAgreementValue(OpponentLongTermIdx[randNum], nCurrentTurn);
						m_dOppSelectedValue = agentOpponentLongTerm.getAgreementValue(OpponentLongTermIdx[randNum], nCurrentTurn);
						m_sAgreement = m_CurrentAgentType.getAgreementStr(OpponentLongTermIdx[randNum]);
						bw.println("Agreement: " + m_sAgreement);
						bw.println("QO Value: " + m_dQOValue);
						bw.println("Agent Value: " + m_dAgentSelectedValue);
						bw.println("Opponent Value: " + m_dOppSelectedValue);
						
						sSelectedOpponent = "Long";
					}
				}
				if (bCalcOpponentShortTerm)
				{
					if (dOpponentShortTermQOValue[randNum] * dBelievedTypeOpponentShortTerm > m_dQOValue)
					{
						bw.println("~~~~~~~~~~SELECTED SHORT~~~~~~~~~~");
						m_dQOValue = dOpponentShortTermQOValue[randNum] * dBelievedTypeOpponentShortTerm;
						m_dAgentSelectedValue = m_CurrentAgentType.getAgreementValue(OpponentShortTermIdx[randNum], nCurrentTurn);
						m_dOppSelectedValue = agentOpponentShortTerm.getAgreementValue(OpponentShortTermIdx[randNum], nCurrentTurn);
						m_sAgreement = m_CurrentAgentType.getAgreementStr(OpponentShortTermIdx[randNum]);
						bw.println("Agreement: " + m_sAgreement);
						bw.println("QO Value: " + m_dQOValue);
						bw.println("Agent Value: " + m_dAgentSelectedValue);
						bw.println("Opponent Value: " + m_dOppSelectedValue);
						
						sSelectedOpponent = "Short";
					}
				}
				
				sSelectedOffer = "Agreement: " + m_sAgreement + "\n" +
				"QO Value: " + m_dQOValue + "\n" +
				"Agent Value: " + m_dAgentSelectedValue + "\n" +
				"Opponent Value: " + m_dOppSelectedValue;
				
				System.err.println("~~~~~~~~~~~~~~~~~~~~~~~~~~");
				System.err.println("Will Send Message for Opponent: " + sSelectedOpponent);
				System.err.println(sSelectedOffer);
				System.err.println("~~~~~~~~~~~~~~~~~~~~~~~~~~");

			} // end if-else - calculate for current/next turn
			
			// write probabilities to file
			bw.println("-----------------Final Probabilities-------------");
			bw.println("Short: " + dBelievedTypeOpponentShortTerm);
			bw.println("Long: " + dBelievedTypeOpponentLongTerm);
			bw.println("Comp: " + dBelievedTypeOpponentCompromise);
			
			bw.close();
			
			} catch (Exception e) {
				System.out.println("Error opening QO log file [QAgentsCore::OfferAgainstOpponent(592)]");
				System.err.println("Error opening QO log file [QAgentsCore::OfferAgainstOpponent(592)]");
				e.printStackTrace();
			}
		}
		
		public double getSelectedQOAgreementValue()
		{
			return m_dQOValue;
		}
		
		public double getSelectedEquilibriumAgreementValue()
		{
			return m_dEquilibriumValue;
		}
		
		public String getSelectedQOAgreementStr()
		{
			return m_sAgreement;
		}
		
		public String getSelectedEquilibriumAgreementStr()
		{
			return m_sEquilibriumAgreement;
		}
		
		public double getNextTurnAgentQOUtilityValue()
		{
			return m_dAgentSelectedNextTurnValue;
		}
		
		public double getNextTurnAgentEquilibriumUtilityValue()
		{
			//return m_dNextTurnEquilibriumValue;
			//22-09-05
			return m_dNextTurnCurrentAgentValueForEqOffer;
		}

		public String getNextTurnQOAgreement()
		{
			return m_sNextTurnAgreement;
		}
		
		public String getNextTurnEquilibriumAgreement()
		{
			return m_sNextTurnEquilibriumAgreement;
		}
		
		public double getNextTurnOpponentQOUtilityValue()
		{
			return m_dOppSelectedNextTurnValue;
		}
		
		public int getNextTurnOpponentType()
		{
			return m_nNextTurnOppType;
		}

		// PRE-CONDITION: m_CurrentAgentType should be updated for the current turn
		public void calculateEquilibrium(QAgentType agentType, int nMaxTurns, boolean bCalculateForAllAgents, boolean bCalcForNextTurn, int nCurrentTurn)
		{
			m_dEquilibriumValue = QAgentType.VERY_SMALL_NUMBER;
			m_dNextTurnEquilibriumValue = QAgentType.VERY_SMALL_NUMBER;
			
			if (bCalcForNextTurn)
				m_CurrentAgentNextTurnType = agentType;
			else
				m_CurrentAgentType = agentType;
	
			if (m_CurrentAgentType.isTypeOf(QAgentType.ZIMBABWE_TYPE))
			{
				QAgentType engCompromise = null;
				if (bCalcForNextTurn)
					engCompromise = getEnglandCompromiseNextTurnType();
				else
					engCompromise = getEnglandCompromiseType();
				
//				if (getDoesOpponentEnd())
					calculateEquilibrium(engCompromise, true, nMaxTurns, COMPROMISE_TYPE_IDX, FIRST_EQUILIBRIUM, bCalculateForAllAgents, bCalcForNextTurn, nCurrentTurn);
//				else
//				{
//					calculateEquilibrium(engCompromise, true, nMaxTurns, COMPROMISE_TYPE_IDX, FIRST_EQUILIBRIUM, bCalculateForAllAgents, bCalcForNextTurn, nCurrentTurn);
//					calculateEquilibrium(engCompromise, false, nMaxTurns, COMPROMISE_TYPE_IDX, SECOND_EQUILIBRIUM, bCalculateForAllAgents, bCalcForNextTurn, nCurrentTurn);
//				}
				
				QAgentType engLongTerm = null;
				if (bCalcForNextTurn)
					engLongTerm = getEnglandLongTermNextTurnType();
				else
					engLongTerm = getEnglandLongTermType();
				
//				if (getDoesOpponentEnd())
					calculateEquilibrium(engLongTerm, true, nMaxTurns, LONG_TERM_TYPE_IDX, FIRST_EQUILIBRIUM, bCalculateForAllAgents, bCalcForNextTurn, nCurrentTurn);
//				else {
//					calculateEquilibrium(engLongTerm, true, nMaxTurns, LONG_TERM_TYPE_IDX, FIRST_EQUILIBRIUM, bCalculateForAllAgents, bCalcForNextTurn, nCurrentTurn);
//					calculateEquilibrium(engLongTerm, false, nMaxTurns, LONG_TERM_TYPE_IDX, SECOND_EQUILIBRIUM, bCalculateForAllAgents, bCalcForNextTurn, nCurrentTurn);
//				}
				
				// this is the real opponent
				QAgentType engShortTerm = null;
				if (bCalcForNextTurn)
					engShortTerm = getEnglandShortTermNextTurnType();
				else
					engShortTerm = getEnglandShortTermType();

//				if (getDoesOpponentEnd())
					calculateEquilibrium(engShortTerm, true, nMaxTurns, SHORT_TERM_TYPE_IDX, FIRST_EQUILIBRIUM, bCalculateForAllAgents, bCalcForNextTurn, nCurrentTurn);
//				else {
//					calculateEquilibrium(engShortTerm, true, nMaxTurns, SHORT_TERM_TYPE_IDX, FIRST_EQUILIBRIUM, bCalculateForAllAgents, bCalcForNextTurn, nCurrentTurn);
//					calculateEquilibrium(engShortTerm, false, nMaxTurns, SHORT_TERM_TYPE_IDX, SECOND_EQUILIBRIUM, bCalculateForAllAgents, bCalcForNextTurn, nCurrentTurn);
//				}
			}
			else if (m_CurrentAgentType.isTypeOf(QAgentType.ENGLAND_TYPE))
			{
				QAgentType zimCompromise = null;
				if (bCalcForNextTurn)
					zimCompromise = getZimbabweCompromiseNextTurnType();
				else
					zimCompromise = getZimbabweCompromiseType();

//				if (getDoesOpponentEnd())
					calculateEquilibrium(zimCompromise, true, nMaxTurns, COMPROMISE_TYPE_IDX, FIRST_EQUILIBRIUM, bCalculateForAllAgents, bCalcForNextTurn, nCurrentTurn);
//				else {
//					calculateEquilibrium(zimCompromise, true, nMaxTurns, COMPROMISE_TYPE_IDX, FIRST_EQUILIBRIUM, bCalculateForAllAgents, bCalcForNextTurn, nCurrentTurn);
//					calculateEquilibrium(zimCompromise, false, nMaxTurns, COMPROMISE_TYPE_IDX, SECOND_EQUILIBRIUM, bCalculateForAllAgents, bCalcForNextTurn, nCurrentTurn);
//				}
				
				QAgentType zimLongTerm = null;
				if (bCalcForNextTurn)
					zimLongTerm = getZimbabweLongTermNextTurnType();
				else
					zimLongTerm = getZimbabweLongTermType();

//				if (getDoesOpponentEnd())
					calculateEquilibrium(zimLongTerm, true, nMaxTurns, LONG_TERM_TYPE_IDX, FIRST_EQUILIBRIUM, bCalculateForAllAgents, bCalcForNextTurn, nCurrentTurn);
//				else {
//					calculateEquilibrium(zimLongTerm, true, nMaxTurns, LONG_TERM_TYPE_IDX, FIRST_EQUILIBRIUM, bCalculateForAllAgents, bCalcForNextTurn, nCurrentTurn);
//					calculateEquilibrium(zimLongTerm, false, nMaxTurns, LONG_TERM_TYPE_IDX, SECOND_EQUILIBRIUM, bCalculateForAllAgents, bCalcForNextTurn, nCurrentTurn);
//				}
				
				// this is the real opponent
				QAgentType zimShortTerm = null;
				if (bCalcForNextTurn)
					zimShortTerm = getZimbabweShortTermNextTurnType();
				else
					zimShortTerm = getZimbabweShortTermType();

//				if (getDoesOpponentEnd())
					calculateEquilibrium(zimShortTerm, true, nMaxTurns, SHORT_TERM_TYPE_IDX, FIRST_EQUILIBRIUM, bCalculateForAllAgents, bCalcForNextTurn, nCurrentTurn);
//				else {
//					calculateEquilibrium(zimShortTerm, true, nMaxTurns, SHORT_TERM_TYPE_IDX, FIRST_EQUILIBRIUM, bCalculateForAllAgents, bCalcForNextTurn, nCurrentTurn);
//					calculateEquilibrium(zimShortTerm, false, nMaxTurns, SHORT_TERM_TYPE_IDX, SECOND_EQUILIBRIUM, bCalculateForAllAgents, bCalcForNextTurn, nCurrentTurn);
//				}
			}
			else
			{					
				System.out.println("[QO]Agent type is unknown [QAgentsCore::calculateEquilibrium(658");
				System.err.println("[QO]Agent type is unknown [QAgentsCore::calculateEquilibrium(658)]");
				return;
			}
			
			// calculate the selected equilbrium agreement and value
			if (bCalculateForAllAgents)
			{
				// choose the best value from all possible values
				ArrayList<Double> dSortedValuesList = new ArrayList<Double>();
				ArrayList<String> sSortedValuesList = new ArrayList<String>();
				
				double dCurrentValue = QAgentType.VERY_SMALL_NUMBER;
				boolean bFoundInd = false;
				int nInsertionPoint = 0;
				for (int i = 0; i < AGENT_TYPES_NUM; ++i)
				{
					// insert values in a sorted way
					
					// first values
					bFoundInd = false;
					for (int ind = 0; ind < dSortedValuesList.size() && !bFoundInd; ++ind)
					{
						dCurrentValue = ((Double)dSortedValuesList.get(ind)).doubleValue();
						
						if (bCalcForNextTurn)
						{
							if (m_dFirstEquilibriumValueNextTurn[i] < dCurrentValue)
							{
								bFoundInd = true;
								nInsertionPoint = ind;
							}
						}
						else
						{
							if (m_dFirstEquilibriumValue[i] < dCurrentValue)
							{
								bFoundInd = true;
								nInsertionPoint = ind;
							}
						}
					}
					if (bFoundInd)
					{
						if (bCalcForNextTurn)
						{
							dSortedValuesList.add(nInsertionPoint, new Double(m_dFirstEquilibriumValueNextTurn[i]));
							sSortedValuesList.add(nInsertionPoint, m_sFirstEquilibriumAgreementNextTurn[i]);
						}
						else
						{
							dSortedValuesList.add(nInsertionPoint, new Double(m_dFirstEquilibriumValue[i]));
							sSortedValuesList.add(nInsertionPoint, m_sFirstEquilibriumAgreement[i]);
						}
					}
					else 
					{
						if (bCalcForNextTurn)
						{
							dSortedValuesList.add(new Double(m_dFirstEquilibriumValueNextTurn[i]));
							sSortedValuesList.add(m_sFirstEquilibriumAgreementNextTurn[i]);
						}
						else
						{
							dSortedValuesList.add(new Double(m_dFirstEquilibriumValue[i]));
							sSortedValuesList.add(m_sFirstEquilibriumAgreement[i]);
						}
					}

					// second values
					bFoundInd = false;
					for (int ind = 0; ind < dSortedValuesList.size() && !bFoundInd; ++ind)
					{
						dCurrentValue = ((Double)dSortedValuesList.get(ind)).doubleValue();
						
						if (bCalcForNextTurn)
						{
							if (m_dSecondEquilibriumValueNextTurn[i] < dCurrentValue)
							{
								bFoundInd = true;
								nInsertionPoint = ind;
							}
						}
						else
						{
							if (m_dSecondEquilibriumValue[i] < dCurrentValue)
							{
								bFoundInd = true;
								nInsertionPoint = ind;
							}
						}
					}
					if (bFoundInd)
					{
						if (bCalcForNextTurn)
						{
							dSortedValuesList.add(nInsertionPoint, new Double(m_dSecondEquilibriumValueNextTurn[i]));
							sSortedValuesList.add(nInsertionPoint, m_sSecondEquilibriumAgreementNextTurn[i]);
						}
						else
						{
							dSortedValuesList.add(nInsertionPoint, new Double(m_dSecondEquilibriumValue[i]));
							sSortedValuesList.add(nInsertionPoint, m_sSecondEquilibriumAgreement[i]);
						}
					}
					else
					{
						if (bCalcForNextTurn)
						{
							dSortedValuesList.add(new Double(m_dSecondEquilibriumValueNextTurn[i]));
							sSortedValuesList.add(m_sSecondEquilibriumAgreementNextTurn[i]);
						}
						else
						{
							dSortedValuesList.add(new Double(m_dSecondEquilibriumValue[i]));
							sSortedValuesList.add(m_sSecondEquilibriumAgreement[i]);
						}
					}
				}
				
				if (bCalcForNextTurn)
				{
					m_dNextTurnEquilibriumValue = ((Double)dSortedValuesList.get(dSortedValuesList.size() - 1)).doubleValue();
					m_sNextTurnEquilibriumAgreement = (String)sSortedValuesList.get(sSortedValuesList.size() - 1);
					m_dNextTurnCurrentAgentValueForEqOffer = agentType.getAgreementValue(agentType.getAgreementIndices(m_sNextTurnEquilibriumAgreement), nCurrentTurn+1);
				}
				else
				{
					m_dEquilibriumValue = ((Double)dSortedValuesList.get(dSortedValuesList.size() - 1)).doubleValue();
					m_sEquilibriumAgreement = (String)sSortedValuesList.get(sSortedValuesList.size() - 1);
				}
			}
			else // calculate only for a predefined agent (short-term)
			{
				if (bCalcForNextTurn)
				{
					if (m_dFirstEquilibriumValueNextTurn[SHORT_TERM_TYPE_IDX] > m_dSecondEquilibriumValueNextTurn[SHORT_TERM_TYPE_IDX])
					{
						m_dNextTurnEquilibriumValue = m_dFirstEquilibriumValueNextTurn[SHORT_TERM_TYPE_IDX];
						m_sNextTurnEquilibriumAgreement = m_sFirstEquilibriumAgreementNextTurn[SHORT_TERM_TYPE_IDX];
						m_dNextTurnCurrentAgentValueForEqOffer = agentType.getAgreementValue(agentType.getAgreementIndices(m_sNextTurnEquilibriumAgreement), nCurrentTurn+1);
					}
					else if (m_dFirstEquilibriumValueNextTurn[SHORT_TERM_TYPE_IDX] < m_dSecondEquilibriumValueNextTurn[SHORT_TERM_TYPE_IDX])
					{
						m_dNextTurnEquilibriumValue = m_dSecondEquilibriumValueNextTurn[SHORT_TERM_TYPE_IDX];
						m_sNextTurnEquilibriumAgreement = m_sSecondEquilibriumAgreementNextTurn[SHORT_TERM_TYPE_IDX];
						m_dNextTurnCurrentAgentValueForEqOffer = agentType.getAgreementValue(agentType.getAgreementIndices(m_sNextTurnEquilibriumAgreement), nCurrentTurn+1);
					}
					else // equals
					{
						// flip a coin
						Random generator = new Random();
						double dRandNum = generator.nextDouble();		
						if (dRandNum <= 0.5)
						{
							m_dNextTurnEquilibriumValue = m_dFirstEquilibriumValueNextTurn[SHORT_TERM_TYPE_IDX];
							m_sNextTurnEquilibriumAgreement = m_sFirstEquilibriumAgreementNextTurn[SHORT_TERM_TYPE_IDX];
							m_dNextTurnCurrentAgentValueForEqOffer = agentType.getAgreementValue(agentType.getAgreementIndices(m_sNextTurnEquilibriumAgreement), nCurrentTurn+1);
						}
						else
						{
							m_dNextTurnEquilibriumValue = m_dSecondEquilibriumValueNextTurn[SHORT_TERM_TYPE_IDX];
							m_sNextTurnEquilibriumAgreement = m_sSecondEquilibriumAgreementNextTurn[SHORT_TERM_TYPE_IDX];
							m_dNextTurnCurrentAgentValueForEqOffer = agentType.getAgreementValue(agentType.getAgreementIndices(m_sNextTurnEquilibriumAgreement), nCurrentTurn+1);
						}
					}					
				}
				else
				{
					if (m_dFirstEquilibriumValue[SHORT_TERM_TYPE_IDX] > m_dSecondEquilibriumValue[SHORT_TERM_TYPE_IDX])
					{
						m_dEquilibriumValue = m_dFirstEquilibriumValue[SHORT_TERM_TYPE_IDX];
						m_sEquilibriumAgreement = m_sFirstEquilibriumAgreement[SHORT_TERM_TYPE_IDX];
					}
					else if (m_dFirstEquilibriumValue[SHORT_TERM_TYPE_IDX] < m_dSecondEquilibriumValue[SHORT_TERM_TYPE_IDX])
					{
						m_dEquilibriumValue = m_dSecondEquilibriumValue[SHORT_TERM_TYPE_IDX];
						m_sEquilibriumAgreement = m_sSecondEquilibriumAgreement[SHORT_TERM_TYPE_IDX];
					}
					else // equals
					{
						// flip a coin
						Random generator = new Random();
						double dRandNum = generator.nextDouble();		
						if (dRandNum <= 0.5)
						{
							m_dEquilibriumValue = m_dFirstEquilibriumValue[SHORT_TERM_TYPE_IDX];
							m_sEquilibriumAgreement = m_sFirstEquilibriumAgreement[SHORT_TERM_TYPE_IDX];
						}
						else
						{
							m_dEquilibriumValue = m_dSecondEquilibriumValue[SHORT_TERM_TYPE_IDX];
							m_sEquilibriumAgreement = m_sSecondEquilibriumAgreement[SHORT_TERM_TYPE_IDX];
						}
					}
				}
			} // end if-else: calculate for all agents or only one
		} // end calculateEquilibrium

		public void calculateEquilibrium(QAgentType oppAgentType, boolean bOppEnds, int nMaxTurns, int nAgentType, int nEquilibriumNum, boolean bCalculateForAllAgents, boolean bCalcForNextTurn, int nCurrentTurn)
		{	
			QCombinedAgreement ca = new QCombinedAgreement();
//			ca.m_dAgentAgreementValue = dCurrentRecvValue;
//			ca.m_dOpponentAgreementValue = dCurrentOppValue;
//			
			QAgentType currentAgentType = null;
			
			if (bCalcForNextTurn)
				currentAgentType = m_CurrentAgentNextTurnType;
			else
				currentAgentType = m_CurrentAgentType;
			
//			if (bCalcForNextTurn)
			{
				if (nCurrentTurn >= nMaxTurns)
					nCurrentTurn = nMaxTurns - 1;
			}
			
			// get expected utility for best agreement at time nMaxTurns - 1
			double dAgreements[] = new double[2];
			dAgreements[0] = QAgentType.VERY_SMALL_NUMBER;
			dAgreements[1] = QAgentType.VERY_SMALL_NUMBER;
			getBothBestAgreementsAtTime(currentAgentType,  oppAgentType, nMaxTurns -1, dAgreements, ca);
			
			for (int t = nMaxTurns - 2; t >= nCurrentTurn; --t)
			{
				getBothBestAgreementsAtTime(currentAgentType, oppAgentType, t, dAgreements, ca);
			}
			
			// save agreement values
			if (nEquilibriumNum == FIRST_EQUILIBRIUM)
			{
				if (bCalcForNextTurn)
				{
					m_sFirstEquilibriumAgreementNextTurn[nAgentType] = ca.m_sAgreement;
					
					int tempCurrentAgreementIdx[] = new int[QAgentType.MAX_ISSUES];
					tempCurrentAgreementIdx = currentAgentType.getAgreementIndices(ca.m_sAgreement);
					double dCurrentAgentValue = currentAgentType.getAgreementValue(tempCurrentAgreementIdx, nCurrentTurn);
					/*if (bOppEnds)
						m_dFirstEquilibriumValueNextTurn[nAgentType] = ca.m_dAgentAgreementValue;
					else
						m_dFirstEquilibriumValueNextTurn[nAgentType] = ca.m_dOpponentAgreementValue;
					*/
					
					m_dFirstEquilibriumValueNextTurn[nAgentType] = dCurrentAgentValue;
					
					if (bCalculateForAllAgents)
						m_dFirstEquilibriumValueNextTurn[nAgentType] *= oppAgentType.getTypeProbability();
				}
				else
				{
					m_sFirstEquilibriumAgreement[nAgentType] = ca.m_sAgreement;
					
					int tempCurrentAgreementIdx[] = new int[QAgentType.MAX_ISSUES];
					tempCurrentAgreementIdx = currentAgentType.getAgreementIndices(ca.m_sAgreement);
					double dCurrentAgentValue = currentAgentType.getAgreementValue(tempCurrentAgreementIdx, nCurrentTurn);
					/*if (bOppEnds)
						m_dFirstEquilibriumValue[nAgentType] = ca.m_dAgentAgreementValue;
					else
						m_dFirstEquilibriumValue[nAgentType] = ca.m_dOpponentAgreementValue;
					*/
					m_dFirstEquilibriumValue[nAgentType] = dCurrentAgentValue;
					if (bCalculateForAllAgents)
						m_dFirstEquilibriumValue[nAgentType] *= oppAgentType.getTypeProbability();
				}
			}
			else if (nEquilibriumNum == SECOND_EQUILIBRIUM)
			{
				if (bCalcForNextTurn)
				{
					m_sSecondEquilibriumAgreementNextTurn[nAgentType] = ca.m_sAgreement;
					
					int tempCurrentAgreementIdx[] = new int[QAgentType.MAX_ISSUES];
					tempCurrentAgreementIdx = currentAgentType.getAgreementIndices(ca.m_sAgreement);
					double dCurrentAgentValue = currentAgentType.getAgreementValue(tempCurrentAgreementIdx, nCurrentTurn);
					/*if (bOppEnds)
						m_dSecondEquilibriumValueNextTurn[nAgentType] = ca.m_dAgentAgreementValue;
					else
						m_dSecondEquilibriumValueNextTurn[nAgentType] = ca.m_dOpponentAgreementValue;
					*/
					m_dSecondEquilibriumValueNextTurn[nAgentType] = dCurrentAgentValue;
					
					if (bCalculateForAllAgents)
						m_dSecondEquilibriumValueNextTurn[nAgentType] *= oppAgentType.getTypeProbability();
				}
				else
				{
					m_sSecondEquilibriumAgreement[nAgentType] = ca.m_sAgreement;
					
					int tempCurrentAgreementIdx[] = new int[QAgentType.MAX_ISSUES];
					tempCurrentAgreementIdx = currentAgentType.getAgreementIndices(ca.m_sAgreement);
					double dCurrentAgentValue = currentAgentType.getAgreementValue(tempCurrentAgreementIdx, nCurrentTurn);
					/*if (bOppEnds)
						m_dSecondEquilibriumValue[nAgentType] = ca.m_dAgentAgreementValue;
					else
						m_dSecondEquilibriumValue[nAgentType] = ca.m_dOpponentAgreementValue;
					*/
					m_dSecondEquilibriumValue[nAgentType] = dCurrentAgentValue;
					
					if (bCalculateForAllAgents)
						m_dSecondEquilibriumValue[nAgentType] *= oppAgentType.getTypeProbability();
				}
			}
/*			
			m_out.println("Perfect Equilibrium Agreement:");
			if (bOppEnds)
				m_out.println("England Ends--- Me: " + ca.m_dAgentAgreementValue + " Opp: " + ca.m_dOpponentAgreementValue + "\n" + ca.m_sAgreement);
			else
				m_out.println("Zimbabwe Ends--- Me: " + ca.m_dOpponentAgreementValue + " Opp: " + ca.m_dAgentAgreementValue + "\n" + ca.m_sAgreement);
*/
		}
		
		/**
		 * @param i
		 * @param agreements
		 */
		private void getBothBestAgreementsAtTime(QAgentType first, QAgentType second, int nCurrentTurn, double[] agreements, QCombinedAgreement ca) {
			double dFirstValue = QAgentType.VERY_SMALL_NUMBER;
			double dSecondValue = QAgentType.VERY_SMALL_NUMBER;
			
			double dFirstUtilityFromSecondBest = QAgentType.VERY_SMALL_NUMBER;
			double dSecondUtilityFromFirstBest = QAgentType.VERY_SMALL_NUMBER;
		//	double dFirstUtilityFromFirstBest = agreements[0];
		//	double dSecondUtilityFromSecondBest = agreements[1];
			
		//	double agreementsSum = agreements[0] + agreements[1];
			
		//	agreements[0] = 0.5 * (agreementsSum);
		//	agreements[1] = 0.5 * (agreementsSum);
						
			double initialAgreement0 = agreements[0];
			double initialAgreement1 = agreements[1];

			int nIssuesNum = m_CurrentAgentType.getIssuesNum();
			int nTotalAgreements = m_CurrentAgentType.getTotalAgreements();
					
			int CurrentAgreementIdx[] = new int[nIssuesNum];
			int MaxIssueValues[] = new int[nIssuesNum];
			
			for (int i = 0; i < nIssuesNum; ++i)
			{
				CurrentAgreementIdx[i] = 0;
				MaxIssueValues[i] = m_CurrentAgentType.getMaxIssueValue(i);
			}
			
			for (int i = 0; i < nTotalAgreements; ++i)
			{
				dFirstValue = first.getAgreementValue(CurrentAgreementIdx, nCurrentTurn);
				dSecondValue = second.getAgreementValue(CurrentAgreementIdx, nCurrentTurn);
				
				// if this agreement is better for the opponent than the agreement
				// offered at time t+1
				if (dSecondValue > initialAgreement1)
				{
					if (dFirstValue > agreements[0])
					{
						agreements[0] = dFirstValue;
						dSecondUtilityFromFirstBest = dSecondValue;
						ca.m_sAgreement = first.getAgreementStr(CurrentAgreementIdx);	
					}							
				}
				// receiveMessage issue values indices
				boolean bFinishUpdate = false;
				for (int k = nIssuesNum-1; k >= 0 && !bFinishUpdate; --k)
				{
					if (CurrentAgreementIdx[k]+1 >= MaxIssueValues[k])
					{
						CurrentAgreementIdx[k] = 0;
					}
					else
					{
						CurrentAgreementIdx[k]++;
						bFinishUpdate = true;
					}									
				}
			}

			for (int i = 0; i < nIssuesNum; ++i)
			{
				CurrentAgreementIdx[i] = 0;
				MaxIssueValues[i] = m_CurrentAgentType.getMaxIssueValue(i);
			}

			for (int i = 0; i < nTotalAgreements; ++i)
			{
				dFirstValue = first.getAgreementValue(CurrentAgreementIdx, nCurrentTurn);
				dSecondValue = second.getAgreementValue(CurrentAgreementIdx, nCurrentTurn);
				
				if (dFirstValue > initialAgreement0)
				{
					if (dSecondValue > agreements[1])
					{
						agreements[1] = dSecondValue;
						dFirstUtilityFromSecondBest = dFirstValue;
						
					}							
				}

				// receiveMessage issue values indices
				boolean bFinishUpdate = false;
				for (int k = nIssuesNum-1; k >= 0 && !bFinishUpdate; --k)
				{
					if (CurrentAgreementIdx[k]+1 >= MaxIssueValues[k])
					{
						CurrentAgreementIdx[k] = 0;
					}
					else
					{
						CurrentAgreementIdx[k]++;
						bFinishUpdate = true;
					}									
				}
			}
			
			// receiveMessage values
			if (dFirstUtilityFromSecondBest > QAgentType.VERY_SMALL_NUMBER)
				agreements[0] = 0.5 * (agreements[0] + dFirstUtilityFromSecondBest);
			if (dSecondUtilityFromFirstBest > dFirstUtilityFromSecondBest)
				agreements[1] = 0.5 * (agreements[1] + dSecondUtilityFromFirstBest);
		}


		// return an agreement the offerAgent will offer at time period nCurrentTime
		// the agreement is the best agreement for offerAgent that is highest than fGivenOfferValue
		// and somewhat higher than fGivenRecvValue
		// the new values for the agent and the opponent are saved in the CombinedAgreement parameter
		public void getBestAgreementAtTime(QAgentType offerAgent, QAgentType recvAgent, QCombinedAgreement ca, int nCurrentTurn)
		{
			double dGivenRecvValue = ca.m_dOpponentAgreementValue;
			double dGivenOfferValue = ca.m_dAgentAgreementValue;
			
			double dAgreementValue = dGivenOfferValue, dOfferAgentValue = 0, dRecvAgentValue = 0;

			int nIssuesNum = m_CurrentAgentType.getIssuesNum();
			int nTotalAgreements = m_CurrentAgentType.getTotalAgreements();
					
			int CurrentAgreementIdx[] = new int[nIssuesNum];
			int MaxIssueValues[] = new int[nIssuesNum];
			
			for (int i = 0; i < nIssuesNum; ++i)
			{
				CurrentAgreementIdx[i] = 0;
				MaxIssueValues[i] = m_CurrentAgentType.getMaxIssueValue(i);
			}
			
			for (int i = 0; i < nTotalAgreements; ++i)
			{
				dRecvAgentValue = recvAgent.getAgreementValue(CurrentAgreementIdx, nCurrentTurn);
				dOfferAgentValue = offerAgent.getAgreementValue(CurrentAgreementIdx, nCurrentTurn);
				
				// if this agreement is better for the opponent than the agreement
				// offered at time t+1
				if (dRecvAgentValue > dGivenRecvValue)
				{
					if (dOfferAgentValue > dAgreementValue)
					{
						dAgreementValue = dOfferAgentValue;
									
						ca.m_dOpponentAgreementValue = dRecvAgentValue;
						ca.m_dAgentAgreementValue = dOfferAgentValue;
									
						ca.m_sAgreement = offerAgent.getAgreementStr(CurrentAgreementIdx);	
					}							
				}

				// receiveMessage issue values indices
				boolean bFinishUpdate = false;
				for (int k = nIssuesNum-1; k >= 0 && !bFinishUpdate; --k)
				{
					if (CurrentAgreementIdx[k]+1 >= MaxIssueValues[k])
					{
						CurrentAgreementIdx[k] = 0;
					}
					else
					{
						CurrentAgreementIdx[k]++;
						bFinishUpdate = true;
					}									
				}
			}
		}
	};
	
	
	
	/**
	 * Initializes the agent's core.
	 * Creates the different England types and Zimbabwe types. 
	 */
	public QAgentsCore(String sFileName, String sNow, agents.QOAgent agent)
	{
		m_Agent = agent;
		m_bEquilibriumAgent = false;
		m_sLogFileName = sFileName;
		
		m_CurrentAgentType = null;
		m_CurrentAgentNextTurnType = null;
		
		m_EnglandAgentTypesList = new ArrayList<QAgentType>();
		m_ZimbabweAgentTypesList = new ArrayList<QAgentType>();
		
		m_EnglandAgentTypesNextTurnList = new ArrayList<QAgentType>();
		m_ZimbabweAgentTypesNextTurnList = new ArrayList<QAgentType>();
		
		m_sProbFileName = "logs\\prob" + sNow + ".";

		for (int i = 0; i < AGENT_TYPES_NUM; ++i)
		{
			m_EnglandAgentTypesList.add(i, new QAgentType(m_bEquilibriumAgent));
			m_ZimbabweAgentTypesList.add(i, new QAgentType(m_bEquilibriumAgent));

			m_EnglandAgentTypesNextTurnList.add(i, new QAgentType(m_bEquilibriumAgent));
			m_ZimbabweAgentTypesNextTurnList.add(i, new QAgentType(m_bEquilibriumAgent));
			
		}
		
		createEnglandLongTermType();
		createEnglandShortTermType();
		createEnglandCompromiseType();
		
		createZimbabweLongTermType();
		createZimbabweShortTermType();
		createZimbabweCompromiseType();
	}

	public QAgentsCore(String sFileName, String sNow, boolean bIsEquilibriumAgent, agents.QOAgent agent)
	{
		m_Agent = agent;
		m_bEquilibriumAgent = bIsEquilibriumAgent;
		m_sLogFileName = sFileName;
		
		m_CurrentAgentType = null;
		m_CurrentAgentNextTurnType = null;
		
		m_EnglandAgentTypesList = new ArrayList<QAgentType>();
		m_ZimbabweAgentTypesList = new ArrayList<QAgentType>();
		
		m_EnglandAgentTypesNextTurnList = new ArrayList<QAgentType>();
		m_ZimbabweAgentTypesNextTurnList = new ArrayList<QAgentType>();
		
		m_sProbFileName = "logs\\prob" + sNow + ".";

		for (int i = 0; i < AGENT_TYPES_NUM; ++i)
		{
			m_EnglandAgentTypesList.add(i, new QAgentType(m_bEquilibriumAgent));
			m_ZimbabweAgentTypesList.add(i, new QAgentType(m_bEquilibriumAgent));

			m_EnglandAgentTypesNextTurnList.add(i, new QAgentType(m_bEquilibriumAgent));
			m_ZimbabweAgentTypesNextTurnList.add(i, new QAgentType(m_bEquilibriumAgent));
			
		}
		
		createEnglandLongTermType();
		createEnglandShortTermType();
		createEnglandCompromiseType();
		
		createZimbabweLongTermType();
		createZimbabweShortTermType();
		createZimbabweCompromiseType();
	}

	/**
	 * @return QAgentType - england's long term type
	 */
	public QAgentType getEnglandLongTermType()
	{
		return (QAgentType)m_EnglandAgentTypesList.get(LONG_TERM_TYPE_IDX);
	}

	/**
	 * @return QAgentType - england's short term type
	 */
	public QAgentType getEnglandShortTermType()
	{
		return (QAgentType)m_EnglandAgentTypesList.get(SHORT_TERM_TYPE_IDX);
	}

	/**
	 * @return QAgentType - england's compromise type
	 */
	public QAgentType getEnglandCompromiseType()
	{
		return (QAgentType)m_EnglandAgentTypesList.get(COMPROMISE_TYPE_IDX);
	}

	/**
	 * @return QAgentType - zimbabwe's long term type
	 */
	public QAgentType getZimbabweLongTermType()
	{
		return (QAgentType)m_ZimbabweAgentTypesList.get(LONG_TERM_TYPE_IDX);
	}

	/**
	 * @return QAgentType - zimbabwe's short term type
	 */
	public QAgentType getZimbabweShortTermType()
	{
		return (QAgentType)m_ZimbabweAgentTypesList.get(SHORT_TERM_TYPE_IDX);
	}

	/**
	 * @return QAgentType - zimbabwe's compromise type
	 */
	public QAgentType getZimbabweCompromiseType()
	{
		return (QAgentType)m_ZimbabweAgentTypesList.get(COMPROMISE_TYPE_IDX);
	}

	/**
	 * @return QAgentType - england's long term type
	 */
	public QAgentType getEnglandLongTermNextTurnType()
	{
		return (QAgentType)m_EnglandAgentTypesNextTurnList.get(LONG_TERM_TYPE_IDX);
	}

	/**
	 * @return QAgentType - england's short term type
	 */
	public QAgentType getEnglandShortTermNextTurnType()
	{
		return (QAgentType)m_EnglandAgentTypesNextTurnList.get(SHORT_TERM_TYPE_IDX);
	}

	/**
	 * @return QAgentType - england's compromise type
	 */
	public QAgentType getEnglandCompromiseNextTurnType()
	{
		return (QAgentType)m_EnglandAgentTypesNextTurnList.get(COMPROMISE_TYPE_IDX);
	}

	/**
	 * @return QAgentType - zimbabwe's long term type
	 */
	public QAgentType getZimbabweLongTermNextTurnType()
	{
		return (QAgentType)m_ZimbabweAgentTypesNextTurnList.get(LONG_TERM_TYPE_IDX);
	}

	/**
	 * @return QAgentType - zimbabwe's short term type
	 */
	public QAgentType getZimbabweShortTermNextTurnType()
	{
		return (QAgentType)m_ZimbabweAgentTypesNextTurnList.get(SHORT_TERM_TYPE_IDX);
	}

	/**
	 * @return QAgentType - zimbabwe's compromise type
	 */
	public QAgentType getZimbabweCompromiseNextTurnType()
	{
		return (QAgentType)m_ZimbabweAgentTypesNextTurnList.get(COMPROMISE_TYPE_IDX);
	}

	
	/**
	 * Creates zimbabwe's compromise type from the utility file.
	 * Saves the type in m_ZimbabweAgentTypesList	  
	 */
	private void createZimbabweCompromiseType()
	{
		QAgentType compromiseType = new QAgentType(m_bEquilibriumAgent);
		compromiseType.setAgentType(QAgentType.ZIMBABWE_TYPE);
		
		createAgentTypeFromFile(m_Agent.opponentModels[COMPROMISE_TYPE_IDX], compromiseType);
		
		m_ZimbabweAgentTypesList.set(COMPROMISE_TYPE_IDX, compromiseType);
		
		QAgentType agentTypeNextTurn = compromiseType;
		agentTypeNextTurn.calculateValues(2);
		m_ZimbabweAgentTypesNextTurnList.set(COMPROMISE_TYPE_IDX, agentTypeNextTurn);
	}

	/**
	 * Creates zimbabwe's short term type from the utility file.
	 * Saves the type in m_ZimbabweAgentTypesList	  
	 */
	private void createZimbabweShortTermType()
	{
		QAgentType shortTermType = new QAgentType(m_bEquilibriumAgent);
		shortTermType.setAgentType(QAgentType.ZIMBABWE_TYPE);
		
		createAgentTypeFromFile(m_Agent.opponentModels[SHORT_TERM_TYPE_IDX], shortTermType);
				
		m_ZimbabweAgentTypesList.set(SHORT_TERM_TYPE_IDX, shortTermType);
		
		QAgentType agentTypeNextTurn = shortTermType;
		agentTypeNextTurn.calculateValues(2);
		m_ZimbabweAgentTypesNextTurnList.set(SHORT_TERM_TYPE_IDX, agentTypeNextTurn);
	}

	/**
	 * Creates zimbabwe's long term type from the utility file.
	 * Saves the type in m_ZimbabweAgentTypesList	  
	 */
	private void createZimbabweLongTermType()
	{
		QAgentType longTermType = new QAgentType(m_bEquilibriumAgent);
		longTermType.setAgentType(QAgentType.ZIMBABWE_TYPE);
		
		createAgentTypeFromFile(m_Agent.opponentModels[LONG_TERM_TYPE_IDX], longTermType);
		
		m_ZimbabweAgentTypesList.set(LONG_TERM_TYPE_IDX, longTermType);
		
		QAgentType agentTypeNextTurn = longTermType;
		agentTypeNextTurn.calculateValues(2);
		m_ZimbabweAgentTypesNextTurnList.set(LONG_TERM_TYPE_IDX, agentTypeNextTurn);
	}

	/**
	 * Creates england's comrpomise type from the utility file.
	 * Saves the type in m_EnglandAgentTypesList	  
	 */
	private void createEnglandCompromiseType()
	{
		QAgentType compromiseType = new QAgentType(m_bEquilibriumAgent);
		compromiseType.setAgentType(QAgentType.ENGLAND_TYPE);
	
		createAgentTypeFromFile(m_Agent.opponentModels[COMPROMISE_TYPE_IDX], compromiseType);
		
		
		m_EnglandAgentTypesList.set(COMPROMISE_TYPE_IDX, compromiseType);
		
		QAgentType agentTypeNextTurn = compromiseType;
		agentTypeNextTurn.calculateValues(2);
		m_EnglandAgentTypesNextTurnList.set(COMPROMISE_TYPE_IDX, agentTypeNextTurn);
	}

	/**
	 * Creates england's short term type from the utility file.
	 * Saves the type in m_EnglandAgentTypesList	  
	 */
	private void createEnglandShortTermType()
	{
		QAgentType shortTermType = new QAgentType(m_bEquilibriumAgent);
		shortTermType.setAgentType(QAgentType.ENGLAND_TYPE);
		
		createAgentTypeFromFile(m_Agent.opponentModels[SHORT_TERM_TYPE_IDX], shortTermType);
		
		m_EnglandAgentTypesList.set(SHORT_TERM_TYPE_IDX, shortTermType);

		QAgentType agentTypeNextTurn = shortTermType;
		agentTypeNextTurn.calculateValues(2);
		m_EnglandAgentTypesNextTurnList.set(SHORT_TERM_TYPE_IDX, agentTypeNextTurn);
	}
	
	/**
	 * Creates england's long term type from the utility file.
	 * Saves the type in m_EnglandAgentTypesList	  
	 */
	private void createEnglandLongTermType()
	{
		QAgentType longTermType = new QAgentType(m_bEquilibriumAgent);
		longTermType.setAgentType(QAgentType.ENGLAND_TYPE);
		
		createAgentTypeFromFile(m_Agent.opponentModels[LONG_TERM_TYPE_IDX], longTermType);
		
		m_EnglandAgentTypesList.set(LONG_TERM_TYPE_IDX, longTermType);
		
		QAgentType agentTypeNextTurn = longTermType;
		agentTypeNextTurn.calculateValues(2);
		m_EnglandAgentTypesNextTurnList.set(LONG_TERM_TYPE_IDX, agentTypeNextTurn);
	}
	
	/**
	 * Creates the specific agent type from the file name
	 * Returns the new type in agentType.
	 * @param sFileName - the file name of the agent's type
	 * @param agentType - the returned agent
	 * Note: this function is identical to readUtilityFile in the Client 	  
	 */
	private void createAgentTypeFromFile(AdditiveUtilitySpace utilitySpace, QAgentType agentType)
	{
		//DT: BufferedReader br = null;
		String line;
		
		double dGeneralValues[] = new double[GENERAL_VALUES_NUM];
		
		// init values to default
		dGeneralValues[TIME_EFFECT_IND] = 0;
		dGeneralValues[STATUS_QUO_IND] = QAgentType.VERY_SMALL_NUMBER;
		dGeneralValues[OPT_OUT_IND] = QAgentType.VERY_SMALL_NUMBER;
		
		line = readUtilityDetails(utilitySpace, agentType.m_fullUtility.lstUtilityDetails, dGeneralValues);					
		agentType.m_fullUtility.dTimeEffect = dGeneralValues[TIME_EFFECT_IND];
		agentType.m_fullUtility.dStatusQuoValue = dGeneralValues[STATUS_QUO_IND];
		agentType.m_fullUtility.dOptOutValue = dGeneralValues[OPT_OUT_IND];

		// calculate luce numbers, best agreement and worst agreement at time 0
		agentType.calculateValues(1);
			
		//agentType.printValuesToFile(sFileName);
			
	}		
	
	/**
	 * Read the utility details from the agent's file
	 * @param br - the reader of the file
	 * @param line - the read line 
	 * @param lstUtilityDetails - list of the utility details
	 * @param dGeneralValues - array of the general values
	 * @return line - the new line
	 */
	public String readUtilityDetails(AdditiveUtilitySpace utilitySpace, ArrayList<UtilityDetails> lstUtilityDetails, double dGeneralValues[])
	{
		UtilityDetails utilityDetails = null;


		dGeneralValues[TIME_EFFECT_IND] = -6.;

		dGeneralValues[STATUS_QUO_IND] = 306;

		dGeneralValues[OPT_OUT_IND] = 215;

		utilityDetails = new UtilityDetails();

		// need to add new element to the utilityDetails list

		// get the title
		
		for(genius.core.issue.Issue lTmp : utilitySpace.getDomain().getIssues()) {
		genius.core.issue.IssueDiscrete lIssue = (genius.core.issue.IssueDiscrete )lTmp;
		utilityDetails.sTitle = lIssue.getName();
		// get the attribute name and side
		UtilityIssue utilityIssue = new UtilityIssue();
		utilityIssue.sAttributeName = lIssue.getName();
		utilityIssue.sSide = "Both";
		utilityIssue.dAttributeWeight = utilitySpace.getWeight(lIssue.getNumber())*100;
		for(ValueDiscrete lValue : lIssue.getValues()) {					


			// go over all values
			UtilityValue utilityValue = new UtilityValue();

			utilityValue.sValue = lValue.getValue();

			// get corresponding utility value
			try {
				utilityValue.dUtility = ((EvaluatorDiscrete)(utilitySpace.getEvaluator(lIssue.getNumber()))).getEvaluationNotNormalized(lValue);
			//	++utilityValue.dUtility += NORMALIZE_INCREMENTOR;//TODO: Currently not using normalize incrementor
			}catch (Exception e) {
				e.printStackTrace();
			}

			utilityValue.dTimeEffect = new Double(0);

			utilityIssue.lstUtilityValues.add(utilityValue);
			utilityIssue.sExplanation = lIssue.getDescription();
		} // end for

		utilityDetails.lstUtilityIssues.add(utilityIssue);
	} // end while - reading attributes

	lstUtilityDetails.add(utilityDetails);
	return "";
}
	
	public void updateAgreementsValues(int nTimePeriod)
	{
		QAgentType agentType = null;
		QAgentType agentTypeNextTurn = null;
		for (int i = 0; i < AGENT_TYPES_NUM; ++i)
		{
			agentType = (QAgentType)m_EnglandAgentTypesList.get(i);
			agentType.calculateValues(nTimePeriod);
			m_EnglandAgentTypesList.set(i, agentType);
			
			agentTypeNextTurn = agentType;
			agentTypeNextTurn.calculateValues(nTimePeriod + 1);
			m_EnglandAgentTypesNextTurnList.set(i, agentTypeNextTurn);
			
			agentType = (QAgentType)m_ZimbabweAgentTypesList.get(i);
			agentType.calculateValues(nTimePeriod);
			m_ZimbabweAgentTypesList.set(i, agentType);

			agentTypeNextTurn = agentType;
			agentTypeNextTurn.calculateValues(nTimePeriod + 1);
			m_ZimbabweAgentTypesNextTurnList.set(i, agentTypeNextTurn);
		}
	}
	
	public void initGenerateAgreement(QAgentType agentType)
	{
		m_CurrentAgentType = agentType;
		
		m_GenerateAgreement = new QGenerateAgreement();
	}
	
	public void calculateAgreement(QAgentType agentType, int nCurrentTurn)
	{
		m_GenerateAgreement.calculateAgreement(agentType, nCurrentTurn, false);
	}
	
	public void calculateEquilibriumAgreement(QAgentType agentType, int nMaxTurns, boolean bCalculateForAllAgents, int nCurrentTurn)
	{
		m_GenerateAgreement.calculateEquilibrium(agentType, nMaxTurns, bCalculateForAllAgents, false, nCurrentTurn);
	}
	
	public String getQOAgreement()
	{
		return m_GenerateAgreement.getSelectedQOAgreementStr();
	}
	
	public String getEquilibriumAgreement()
	{
		return m_GenerateAgreement.getSelectedEquilibriumAgreementStr();
	}

	
	public void calculateNextTurnAgreement(QAgentType agentType, int nNextTurn)
	{
		m_GenerateAgreement.calculateAgreement(agentType, nNextTurn, true);
	}
	
	public void calculateNextTurnEquilibriumAgreement(QAgentType agentType, int nMaxTurns, boolean bCalculateForAllAgents, int nNextTurn)
	{
		m_GenerateAgreement.calculateEquilibrium(agentType, nMaxTurns, bCalculateForAllAgents, true, nNextTurn);
	}
	
	public double getNextTurnAgentQOUtilityValue()
	{
		return m_GenerateAgreement.getNextTurnAgentQOUtilityValue();
	}
	
	public double getNextTurnAgentEquilibriumUtilityValue()
	{
		return m_GenerateAgreement.getNextTurnAgentEquilibriumUtilityValue();
	}

	public String getNextTurnAgentQOAgreement()
	{
		return m_GenerateAgreement.getNextTurnQOAgreement();
	}
	
	public String getNextTurnAgentEquilibriumAgreement()
	{
		return m_GenerateAgreement.getNextTurnEquilibriumAgreement();
	}

	public double getNextTurnOpponentQOUtilityValue()
	{
		return m_GenerateAgreement.getNextTurnOpponentQOUtilityValue();
	}

	public QAgentType getNextTurnOpponentType()
	{
		QAgentType opponentNextTurnType = null;
		int nOppType = m_GenerateAgreement.getNextTurnOpponentType();
		
		if (m_CurrentAgentType.isTypeOf(QAgentType.ZIMBABWE_TYPE))
		{
			switch (nOppType)
			{
				case COMPROMISE_TYPE_IDX:
					opponentNextTurnType = getEnglandCompromiseNextTurnType();
					break;
				case LONG_TERM_TYPE_IDX:
					opponentNextTurnType = getEnglandLongTermNextTurnType();
					break;
				case SHORT_TERM_TYPE_IDX:
					opponentNextTurnType = getEnglandShortTermNextTurnType();
					break;
				default:
					System.out.println("[QO]Agent type is unknown [QAgentsCore::getNextTurnOpponentType(1310)]");
					System.err.println("[QO]Agent type is unknown [QAgentsCore::getNextTurnOpponentType(1310)]");
					break;
			}
		}
		else if (m_CurrentAgentType.isTypeOf(QAgentType.ENGLAND_TYPE))
		{
			switch (nOppType)
			{
				case COMPROMISE_TYPE_IDX:
					opponentNextTurnType = getZimbabweCompromiseNextTurnType();
					break;
				case LONG_TERM_TYPE_IDX:
					opponentNextTurnType = getZimbabweLongTermNextTurnType();
					break;
				case SHORT_TERM_TYPE_IDX:
					opponentNextTurnType = getZimbabweShortTermNextTurnType();
					break;
				default:
					System.out.println("[QO]Agent type is unknown [QAgentsCore::getNextTurnOpponentType(1329)]");
					System.err.println("[QO]Agent type is unknown [QAgentsCore::getNextTurnOpponentType(1329)]");
					break;
			}
		}
		
		return opponentNextTurnType;
	}
	
	public void updateOpponentProbability(int CurrentAgreementIdx[], int nCurrentTurn, int nMessageType, int nResponseType)
	{
		if (nResponseType == QMessages.MESSAGE_RECEIVED)
			updateOpponentProbabilityUponMessageReceived(CurrentAgreementIdx, nCurrentTurn, nMessageType);
		else if (nResponseType == QMessages.MESSAGE_REJECTED)
			updateOpponentProbabilityUponMessageRejected(CurrentAgreementIdx, nCurrentTurn, nMessageType);
	}
	
	private void updateOpponentProbabilityUponMessageReceived(int CurrentAgreementIdx[], int nCurrentTurn, int nMessageType)
	{
		QAgentType agentType = null;
		double dPrevTypeProbability = 0;
		double dPrevOfferValue = 0;
		double dPrevOfferProbability = 0;
		double dOfferSum = 0;
		double dUpdatedTypeProbability = 0;
		
		if (m_CurrentAgentType.isTypeOf(QAgentType.ZIMBABWE_TYPE))
		{
			// calculate posteriori proability using Bayes formula:
			// P(type | Ht) = [P(Ht|Type)*P(type)] / P(Ht)
			// where P(Ht) = sigam(i=1 to #types)[P(Ht|Type_i) * P(type_i)]
			// and P(Ht|Type_i) = luce number of Ht (Ht - last agreement)
			// [this is done incrementally after each agreement

			// calculate P(Ht)
			for (int i = 0; i < AGENT_TYPES_NUM; ++i)
			{
				agentType = (QAgentType)m_EnglandAgentTypesList.get(i);
						
				dPrevTypeProbability = agentType.getTypeProbability();
				dPrevOfferValue = agentType.getAgreementValue(CurrentAgreementIdx, nCurrentTurn);
				dPrevOfferProbability = agentType.getAgreementLuceValue(dPrevOfferValue, true);
				
				dOfferSum += (dPrevOfferProbability * dPrevTypeProbability);
			}

			// calculate P(type | Ht) and receiveMessage P(type)
			for (int i = 0; i < AGENT_TYPES_NUM; ++i)
			{
				agentType = (QAgentType)m_EnglandAgentTypesList.get(i);
						
				dPrevTypeProbability = agentType.getTypeProbability();
				dPrevOfferValue = agentType.getAgreementValue(CurrentAgreementIdx, nCurrentTurn);
				dPrevOfferProbability = agentType.getAgreementLuceValue(dPrevOfferValue, true);
				
				dUpdatedTypeProbability = (dPrevOfferProbability * dPrevTypeProbability) / dOfferSum;
				
				System.err.println("%%%%%%%%%%%%%%%%%%%%%%%");
				System.err.println("PREV = " + dPrevTypeProbability + ", UP = " + dUpdatedTypeProbability);
				
				agentType.setTypeProbability(dUpdatedTypeProbability);
			
				m_EnglandAgentTypesList.set(i, agentType);
			}
			
			PrintWriter pw;
			try {
				pw = new PrintWriter(new FileWriter(m_sProbFileName + "Eng.txt", true));
				pw.println(getEnglandProbabilitiesStr());
				pw.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		else if (m_CurrentAgentType.isTypeOf(QAgentType.ENGLAND_TYPE))
		{
			// calculate posteriori proability using Bayes formula:
			// P(type | Ht) = [P(Ht|Type)*P(type)] / P(Ht)
			// where P(Ht) = sigma(i=1 to #types)[P(Ht|Type_i) * P(type_i)]
			// and P(Ht|Type_i) = luce number of Ht (Ht - last agreement)
			// [this is done incrementally after each agreement

			// calculate P(Ht)
			for (int i = 0; i < AGENT_TYPES_NUM; ++i)
			{
				agentType = (QAgentType)m_ZimbabweAgentTypesList.get(i);
						
				dPrevTypeProbability = agentType.getTypeProbability();
				dPrevOfferValue = agentType.getAgreementValue(CurrentAgreementIdx, nCurrentTurn);
				dPrevOfferProbability = agentType.getAgreementLuceValue(dPrevOfferValue, true);
				
				dOfferSum += (dPrevOfferProbability * dPrevTypeProbability);
			}

			// calculate P(type | Ht) and receiveMessage P(type)
			for (int i = 0; i < AGENT_TYPES_NUM; ++i)
			{
				agentType = (QAgentType)m_ZimbabweAgentTypesList.get(i);
						
				dPrevTypeProbability = agentType.getTypeProbability();
				dPrevOfferValue = agentType.getAgreementValue(CurrentAgreementIdx, nCurrentTurn);
				dPrevOfferProbability = agentType.getAgreementLuceValue(dPrevOfferValue, true);
				
				dUpdatedTypeProbability = (dPrevOfferProbability * dPrevTypeProbability) / dOfferSum;
				
				System.err.println("%%%%%%%%%%%%%%%%%%%%%%%");
				System.err.println("PREV = " + dPrevTypeProbability + ", UP = " + dUpdatedTypeProbability);
				
				agentType.setTypeProbability(dUpdatedTypeProbability);
			
				m_ZimbabweAgentTypesList.set(i, agentType);
			}
			
			PrintWriter pw;
			try {
				pw = new PrintWriter(new FileWriter(m_sProbFileName + "Zim.txt", true));
				pw.println(getZimbabweProbabilitiesStr());
				pw.close();
			} catch (IOException e) {
				System.out.println("Error opening QO prob file [QAgentsCore::UpdateOpponentProbabilityUponMessageReceived(1423)]");
				System.err.println("Error opening QO prob file [QAgentsCore::UpdateOpponentProbabilityUponMessageReceived(1423)]");
				e.printStackTrace();
			}
		} // end if-else checking the agent's type
	}
	
	private void updateOpponentProbabilityUponMessageRejected(int CurrentAgreementIdx[], int nCurrentTurn, int nMessageType)
	{
		QAgentType agentType = null;
		double dPrevTypeProbability = 0;
		double dPrevOfferProbability = 0;
		double dOfferSum = 0;
		double dUpdatedTypeProbability = 0;
		double dAgentOfferSum = 0;
		
		String sRejectedMsg = m_CurrentAgentType.getAgreementStr(CurrentAgreementIdx);
		
		if (m_CurrentAgentType.isTypeOf(QAgentType.ZIMBABWE_TYPE))
		{
			// calculate posteriori proability using Bayes formula:
			// P(type | Ht) = [P(Ht|Type)*P(type)] / P(Ht)
			// where P(Ht) = sigma(i=1 to #types)[P(Ht|Type_i) * P(type_i)]
			// and P(Ht|Type_i) = luce number of Ht (Ht - last agreement)
			// [this is done incrementally after each agreement

			// calculate P(Ht)
			for (int i = 0; i < AGENT_TYPES_NUM; ++i)
			{
				agentType = (QAgentType)m_EnglandAgentTypesList.get(i);
						
				dOfferSum += agentType.calcRejectionProbabilities(sRejectedMsg, nCurrentTurn);
			}

			// calculate P(type | Ht) and receiveMessage P(type)
			for (int i = 0; i < AGENT_TYPES_NUM; ++i)
			{
				agentType = (QAgentType)m_EnglandAgentTypesList.get(i);

				dPrevTypeProbability = agentType.getTypeProbability();
				
				dAgentOfferSum = agentType.calcRejectionProbabilities(sRejectedMsg, nCurrentTurn);

				dUpdatedTypeProbability = (dAgentOfferSum * dPrevTypeProbability) / dOfferSum;
				
				System.err.println("%%%%%%%%%%%%%%%%%%%%%%%");
				System.err.println("PREV = " + dPrevTypeProbability + ", UP = " + dUpdatedTypeProbability);
				
				agentType.setTypeProbability(dUpdatedTypeProbability);
			
				m_EnglandAgentTypesList.set(i, agentType);
			}
			
			PrintWriter pw;
			try {
				pw = new PrintWriter(new FileWriter(m_sProbFileName + "Eng.txt", true));
				pw.println(getEnglandProbabilitiesStr());
				pw.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		else if (m_CurrentAgentType.isTypeOf(QAgentType.ENGLAND_TYPE))
		{
			// calculate posteriori proability using Bayes formula:
			// P(type | Ht) = [P(Ht|Type)*P(type)] / P(Ht)
			// where P(Ht) = sigam(i=1 to #types)[P(Ht|Type_i) * P(type_i)]
			// and P(Ht|Type_i) = luce number of Ht (Ht - last agreement)
			// [this is done incrementally after each agreement

			// calculate P(Ht)
			for (int i = 0; i < AGENT_TYPES_NUM; ++i)
			{
				agentType = (QAgentType)m_ZimbabweAgentTypesList.get(i);
				
				dOfferSum += agentType.calcRejectionProbabilities(sRejectedMsg, nCurrentTurn);
			
				dOfferSum += (dPrevOfferProbability * dPrevTypeProbability);
			}

			// calculate P(type | Ht) and receiveMessage P(type)
			for (int i = 0; i < AGENT_TYPES_NUM; ++i)
			{
				agentType = (QAgentType)m_ZimbabweAgentTypesList.get(i);

				dPrevTypeProbability = agentType.getTypeProbability();
				
				dAgentOfferSum = agentType.calcRejectionProbabilities(sRejectedMsg, nCurrentTurn);

				dUpdatedTypeProbability = (dAgentOfferSum * dPrevTypeProbability) / dOfferSum;
				
				System.err.println("%%%%%%%%%%%%%%%%%%%%%%%");
				System.err.println("PREV = " + dPrevTypeProbability + ", UP = " + dUpdatedTypeProbability);
				
				agentType.setTypeProbability(dUpdatedTypeProbability);
			
				m_ZimbabweAgentTypesList.set(i, agentType);
			}
			
			PrintWriter pw;
			try {
				pw = new PrintWriter(new FileWriter(m_sProbFileName + "Zim.txt", true));
				pw.println(getZimbabweProbabilitiesStr());
				pw.close();
			} catch (IOException e) {
				System.out.println("Error opening QO prob file [QAgentsCore::UpdateOpponentProbabilityUponMessageRejected(1525)]");
				System.err.println("Error opening QO prob file [QAgentsCore::UpdateOpponentProbabilityUponMessageRejected(1525)]");
				e.printStackTrace();
			}
		} // end if-else checking the agent's type
	}

	public String getEnglandProbabilitiesStr()
	{
		String sProbabilitiesStr = "";
		
		QAgentType agentType = null;
		double dAgentProbability = 0;
		
		agentType = (QAgentType)m_EnglandAgentTypesList.get(LONG_TERM_TYPE_IDX);
		dAgentProbability = agentType.getTypeProbability();
		
		sProbabilitiesStr = "EngLong: " + dAgentProbability;
		
		agentType = (QAgentType)m_EnglandAgentTypesList.get(SHORT_TERM_TYPE_IDX);
		dAgentProbability = agentType.getTypeProbability();
		
		sProbabilitiesStr += "; EngShort: " + dAgentProbability;

		agentType = (QAgentType)m_EnglandAgentTypesList.get(COMPROMISE_TYPE_IDX);
		dAgentProbability = agentType.getTypeProbability();
		
		sProbabilitiesStr += "; EngComp: " + dAgentProbability;
		
		return sProbabilitiesStr;
	}

	public String getZimbabweProbabilitiesStr()
	{
		String sProbabilitiesStr = "";
		
		QAgentType agentType = null;
		double dAgentProbability = 0;
		
		agentType = (QAgentType)m_ZimbabweAgentTypesList.get(LONG_TERM_TYPE_IDX);
		dAgentProbability = agentType.getTypeProbability();
		
		sProbabilitiesStr = "ZimLong: " + dAgentProbability;
		
		agentType = (QAgentType)m_ZimbabweAgentTypesList.get(SHORT_TERM_TYPE_IDX);
		dAgentProbability = agentType.getTypeProbability();
		
		sProbabilitiesStr += "; ZimShort: " + dAgentProbability;

		agentType = (QAgentType)m_ZimbabweAgentTypesList.get(COMPROMISE_TYPE_IDX);
		dAgentProbability = agentType.getTypeProbability();
		
		sProbabilitiesStr += "; ZimComp: " + dAgentProbability;
		
		return sProbabilitiesStr;
	}
}
