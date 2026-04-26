package genius.core.protocol;

import java.util.ArrayList;
import java.util.HashMap;

import genius.core.Agent;
import genius.core.Bid;
import genius.core.Domain;
import genius.core.NegotiationEventListener;
import genius.core.actions.Action;
import genius.core.analysis.BidPointTime;
import genius.core.analysis.BidSpace;
import genius.core.qualitymeasures.OpponentModelMeasuresResults;
import genius.core.tournament.TournamentConfiguration;
import genius.core.tournament.VariablesAndValues.AgentParamValue;
import genius.core.tournament.VariablesAndValues.AgentParameterVariable;
import genius.core.utility.AbstractUtilitySpace;
import genius.core.utility.AdditiveUtilitySpace;
import genius.core.xml.SimpleElement;

public abstract class BilateralAtomicNegotiationSession implements Runnable {

	protected Agent agentA;
	protected Agent agentB;
	protected AbstractUtilitySpace spaceA;
	protected AbstractUtilitySpace spaceB;
	protected String agentAname;
	protected String agentBname;
	protected Bid lastBid = null; // the last bid that has been done
	protected Action lastAction = null; // the last action that has been done
										// (also included Accept, etc.)
	protected Protocol protocol;
	protected int finalRound; // 0 during whole negotiation accept at agreement,
								// in which case it is equal to rounds
	protected ArrayList<BidPointTime> fAgentABids;
	protected ArrayList<BidPointTime> fAgentBBids;
	protected BidSpace bidSpace;
	protected HashMap<AgentParameterVariable, AgentParamValue> agentAparams;
	protected HashMap<AgentParameterVariable, AgentParamValue> agentBparams;
	protected OpponentModelMeasuresResults omMeasuresResults = new OpponentModelMeasuresResults();

	ArrayList<NegotiationEventListener> actionEventListener = new ArrayList<NegotiationEventListener>();
	private String log;
	/**
	 * tournamentNumber is the tournament.TournamentNumber, or -1 if this
	 * session is not part of a tournament
	 */
	int tournamentNumber = -1;

	public SimpleElement additionalLog = new SimpleElement("additional_log");

	public BilateralAtomicNegotiationSession(Protocol protocol, Agent agentA, Agent agentB, String agentAname,
			String agentBname, AbstractUtilitySpace spaceA, AbstractUtilitySpace spaceB,
			HashMap<AgentParameterVariable, AgentParamValue> agentAparams,
			HashMap<AgentParameterVariable, AgentParamValue> agentBparams) throws Exception {
		this.protocol = protocol;
		this.agentA = agentA;
		this.agentB = agentB;
		this.agentAname = agentAname;
		this.agentBname = agentBname;
		this.spaceA = spaceA;
		this.spaceB = spaceB;
		if (agentAparams != null)
			this.agentAparams = new HashMap<AgentParameterVariable, AgentParamValue>(agentAparams);
		else
			this.agentAparams = new HashMap<AgentParameterVariable, AgentParamValue>();
		if (agentBparams != null)
			this.agentBparams = new HashMap<AgentParameterVariable, AgentParamValue>(agentBparams);
		else
			this.agentBparams = new HashMap<AgentParameterVariable, AgentParamValue>();

		if (TournamentConfiguration.getBooleanOption("accessPartnerPreferences", false)) {
			agentA.fNegotiation = this;
			agentB.fNegotiation = this;
		}
		fAgentABids = new ArrayList<BidPointTime>();
		fAgentBBids = new ArrayList<BidPointTime>();

		Domain domain = spaceA.getDomain();
		String domainName = "";
		if (domain == null)
			System.err.println("Warning: domain null in " + spaceA.getFileName());
		else
			domainName = domain.getName();

		actionEventListener.addAll(protocol.getNegotiationEventListeners());
	}

	public void addNegotiationEventListener(NegotiationEventListener listener) {
		if (!actionEventListener.contains(listener))
			actionEventListener.add(listener);
	}

	public void removeNegotiationEventListener(NegotiationEventListener listener) {
		if (!actionEventListener.contains(listener))
			actionEventListener.remove(listener);
	}

	public double getOpponentUtility(Agent pAgent, Bid pBid) throws Exception {
		if (pAgent.equals(agentA))
			return spaceB.getUtility(pBid);
		else
			return spaceA.getUtility(pBid);
	}

	/**
	 * 
	 * @param pAgent
	 * @param pIssueID
	 * @return weight of issue in opponent's space. 0 if this is not an
	 *         {@link AdditiveUtilitySpace}
	 * @throws Exception
	 */
	public double getOpponentWeight(Agent pAgent, int pIssueID) throws Exception {
		AbstractUtilitySpace space = pAgent.equals(agentA) ? spaceB : spaceA;
		if (space instanceof AdditiveUtilitySpace) {
			return ((AdditiveUtilitySpace) space).getWeight(pIssueID);
		}
		return 0;
	}

	public void addAdditionalLog(SimpleElement pElem) {
		if (pElem != null)
			additionalLog.addChildElement(pElem);

	}

	public int getTournamentNumber() {
		return tournamentNumber;
	}

	public abstract String getStartingAgent();

	public AbstractUtilitySpace getAgentAUtilitySpace() {
		return spaceA;
	}

	public AbstractUtilitySpace getAgentBUtilitySpace() {
		return spaceB;
	}

}
