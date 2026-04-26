package agents.anac.y2012.MetaAgent;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map.Entry;

import agents.anac.y2010.AgentSmith.AgentSmith;
import agents.anac.y2011.AgentK2.Agent_K2;
import agents.anac.y2011.BramAgent.BRAMAgent;
import agents.anac.y2011.Gahboninho.Gahboninho;
import agents.anac.y2011.HardHeaded.KLH;
import agents.anac.y2011.TheNegotiator.TheNegotiator;
import agents.anac.y2011.ValueModelAgent.ValueModelAgent;
import agents.anac.y2012.MetaAgent.agents.Chameleon.Chameleon;
import agents.anac.y2012.MetaAgent.agents.DNAgent.DNAgent;
import agents.anac.y2012.MetaAgent.agents.GYRL.GYRL;
import agents.anac.y2012.MetaAgent.agents.LYY.LYYAgent;
import agents.anac.y2012.MetaAgent.agents.MrFriendly.MrFriendly;
import agents.anac.y2012.MetaAgent.agents.ShAgent.ShAgent;
import agents.anac.y2012.MetaAgent.agents.SimpleAgentNew.SimpleAgentNew;
import agents.anac.y2012.MetaAgent.agents.WinnerAgent.WinnerAgent2;
import genius.core.Agent;
import genius.core.Domain;
import genius.core.SupportedNegotiationSetting;
import genius.core.actions.Action;
import genius.core.actions.EndNegotiation;
import genius.core.actions.Offer;
import genius.core.issue.Issue;
import genius.core.issue.Objective;
import genius.core.issue.ValueDiscrete;
import genius.core.utility.AdditiveUtilitySpace;
import genius.core.utility.EVALUATORTYPE;
import genius.core.utility.Evaluator;
import genius.core.utility.EvaluatorDiscrete;
import genius.core.utility.EvaluatorInteger;
import genius.core.utility.EvaluatorReal;

public class MetaAgent extends Agent {
	private Agent myAgent;
	private String myAgentName;
	private boolean agentInit;
	private Action actionOfPartner;
	private boolean first;
	double[] params;
	double opponentFirstBid;

	@Override
	public void init() {
		agentInit = false;
		first = false;
		opponentFirstBid = 0;
		Domain d = this.utilitySpace.getDomain();
		myAgentName = "";
	}

	private String[][] getRegression() {
		String[][] reg = new String[18][12];
		reg[0] = "IAMcrazyhaggler,0.077,0,0.137,0.000000000,0.015,0,0,0,1.311,-0.03,0.556"
				.split(",");
		reg[1] = "LYYAgent,-1.22,-0.016,0.23,0.000000000,0.027,0.026,1.042,1.372,1.482,0,0.572"
				.split(",");
		reg[2] = "Chameleon,-2.185,-0.024,0.303,0.000000000,0.017,0.055,2.013,2.184,1.692,0,0.502"
				.split(",");
		reg[3] = "AgentSmith,-2.075,-0.018,0.333,-0.000000410,0,0.05,1.938,2.027,1.428,0,0.565"
				.split(",");
		reg[4] = "GYRL,-1.663,0,0.348,0.000000000,0.029,0.028,1.327,1.668,1.327,0,0.609"
				.split(",");
		reg[5] = "Gahboninho,0.309,-0.024,0.337,0.000000248,0,0,0,0,0.495,0,0.451"
				.split(",");
		reg[6] = "ValueModelAgent,-1.401,-0.022,0.441,-0.000000206,0,0.044,1.328,1.231,1.629,0,0.49"
				.split(",");
		reg[7] = "Nice Tit-for-Tat Agent,-0.018,-0.014,0.422,-0.000000381,0,0,0.198,0,0.549,0,0.529"
				.split(",");
		reg[8] = "HardHeaded,-2.012,-0.035,0.319,0.000000172,0,0.053,2.107,2.136,1.547,0,0.392"
				.split(",");
		reg[9] = "WinnerAgent,-1.741,0,0.248,0.000000000,0.031,0.03,1.428,1.796,1.742,0,0.599"
				.split(",");
		reg[10] = "AgentK2,-1.027,-0.012,0.246,0.000000000,0.019,0.022,1.083,1.127,1.097,0,0.531"
				.split(",");
		reg[11] = "TheNegotiator,-0.159,0.017,0.307,0.000000000,0,0.015,0,0,0.92,0.03,0.665"
				.split(",");
		reg[12] = "IAMhaggler2011,0.256,0,0.166,0.000000224,0.017,-0.011,0,0,0.347,0,0.543"
				.split(",");
		reg[13] = "DNAgent,-0.581,-0.024,0.732,0.000000000,0,0.01,0.61,0,1.141,0,0.237"
				.split(",");
		reg[14] = "BRAMAgent,-0.764,0,0.232,0.000000250,0,0.035,0.842,0.708,0.7,0,0.538"
				.split(",");
		reg[15] = "MrFriendly,-1.721,-0.025,0.194,-0.000000429,0,0.054,1.814,1.711,1.488,0,0.498"
				.split(",");
		reg[16] = "ShAgent,-2.072,-0.021,0,0.000000476,0.069,-0.055,2.336,2.805,1.743,0,0.412"
				.split(",");
		reg[17] = "SimpleAgent,-1.359,0,0.287,-0.000000180,0.028,0.016,1.16,1.524,0.86,0,0.588"
				.split(",");
		return reg;
	}

	private void getDomainParams() {
		params = new double[10];
		Domain d = this.utilitySpace.getDomain();
		List<Issue> a = d.getIssues();

		params[0] = a.size(); // 0 = number of issues.
		params[1] = this.utilitySpace.getDiscountFactor() == 0 ? 1
				: this.utilitySpace.getDiscountFactor(); // 1
															// =
															// discount
															// factor.
		int min = Integer.MAX_VALUE, max = -1, size = 1;

		// params 5+6+7 - Expected Utility of role, Standard deviation of
		// Utility of role, Standard deviation of weights of role
		double EU = 0, stdevU = 0, stdevW = 0, sW = 0, ssW = 0, countW = 0;
		Iterator<Entry<Objective, Evaluator>> issue = ((AdditiveUtilitySpace) utilitySpace)
				.getEvaluators().iterator();
		List<Double> ssWList = new ArrayList<Double>();
		while (issue.hasNext()) { // every issue
			Entry<Objective, Evaluator> entry = issue.next();
			Evaluator e = entry.getValue();
			double weight = e.getWeight();
			countW++;
			sW += weight;
			ssWList.add(weight);
			double tempEU = 0, tempStdevU = 0;
			if (e.getType() == EVALUATORTYPE.DISCRETE) {
				Iterator<ValueDiscrete> v = ((EvaluatorDiscrete) e).getValues()
						.iterator();
				List<Double> s = new ArrayList<Double>();
				double sumU = 0;
				while (v.hasNext()) {
					ValueDiscrete vd = v.next();
					try {
						double val = ((EvaluatorDiscrete) e).getEvaluation(vd);
						s.add(val);
						sumU += val;
					} catch (Exception e1) {
						e1.printStackTrace();
					}
				}
				int currSize = s.size();
				min = (min > currSize) ? currSize : min;
				max = (max < currSize) ? currSize : max;
				size = size * currSize;
				tempEU = sumU / currSize;
				Iterator<Double> valIt = s.iterator();
				while (valIt.hasNext()) {
					tempStdevU += Math.pow(valIt.next() - tempEU, 2);
				}
				tempStdevU = Math.sqrt(tempStdevU / ((double) currSize - 1));
			} else if (e.getType() == EVALUATORTYPE.INTEGER) {
				tempEU = ((double) (((EvaluatorInteger) e).getUpperBound()
						+ ((EvaluatorInteger) e).getLowerBound())) / 2;
				tempStdevU = Math.sqrt((Math
						.pow(((EvaluatorInteger) e).getUpperBound() - tempEU, 2)
						+ Math.pow(
								((EvaluatorInteger) e).getLowerBound() - tempEU,
								2))
						/ 2);
			} else if (e.getType() == EVALUATORTYPE.REAL) {
				tempEU = (((EvaluatorReal) e).getUpperBound()
						+ ((EvaluatorReal) e).getLowerBound()) / 2;
				tempStdevU = Math.sqrt((Math
						.pow(((EvaluatorReal) e).getUpperBound() - tempEU, 2)
						+ Math.pow(((EvaluatorReal) e).getLowerBound() - tempEU,
								2))
						/ 2);
			} else {
				tempEU = 0.5;
				tempStdevU = 0;
			}

			EU += tempEU * weight;
			stdevU += tempStdevU * weight;
		}
		Iterator<Double> wIt = ssWList.iterator();
		double avgW = sW / countW;
		while (wIt.hasNext()) {
			ssW += Math.pow(wIt.next() - avgW, 2);
		}

		stdevW = countW <= 1 ? 0 : Math.sqrt(ssW / (countW - 1));
		params[2] = size;
		params[3] = min;
		params[4] = max;
		params[5] = EU;
		params[6] = stdevU;
		params[7] = stdevW;
		params[8] = first ? 1 : 0;
		params[9] = opponentFirstBid;
	}

	private String getBestAgent() {
		double lambda = 70;
		double sumExp = 0;

		String ans = null;
		getDomainParams();
		String[][] reg = getRegression();
		double maxResult = 0;
		int bestIndex = -1;
		List<String> agents = new ArrayList<String>();
		double[] agentsResults = new double[reg.length];
		for (int k = 0; k < reg.length; k++) {
			agents.add(k, reg[k][0]);
			double regressionResult = 0;
			double[] regression = new double[params.length + 1];

			for (int i = 0; i < regression.length; i++) {
				regression[i] = Double.parseDouble(reg[k][i + 1]);
				if (i > 0) {
					regressionResult += regression[i] * params[i - 1];
				} else
					regressionResult += regression[i];
			}
			if (!(reg[k][0].equalsIgnoreCase("Nice Tit-for-Tat Agent")
					|| reg[k][0].equalsIgnoreCase("IAMhaggler2011")
					|| reg[k][0].equalsIgnoreCase("IAMcrazyhaggler"))) {
				// calculate QRE value, part 1
				agentsResults[k] = Math.exp(regressionResult * lambda);
			} else // 3 agents that doesn't work - will get a 0 value.
			{
				agentsResults[k] = 0;
			}
			sumExp += agentsResults[k];

		}
		double rand = Math.random(), totalValue = 0;
		for (int i = 0; i < agentsResults.length; i++) {
			// calculate QRE value, part 2
			agentsResults[i] = agentsResults[i] / sumExp;
			totalValue += agentsResults[i];
			if (rand < totalValue) { // at the end, total value will be 1 (it's
										// a probability value)
				bestIndex = i;
				break;
			}
		}

		ans = agents.get(bestIndex);
		myAgentName = ans;
		return ans;

	}

	private Agent selectAgent(String name) {
		Agent a = new KLH(); // default, as winner of 2011 contest.
		if (name.equalsIgnoreCase("IAMcrazyhaggler"))
			return a;// /agent doesn't work
		else if (name.equalsIgnoreCase("LYYAgent"))
			return new LYYAgent();
		else if (name.equalsIgnoreCase("Chameleon"))
			return new Chameleon();
		else if (name.equalsIgnoreCase("AgentSmith"))
			return new AgentSmith();
		else if (name.equalsIgnoreCase("GYRL"))
			return new GYRL();
		else if (name.equalsIgnoreCase("Gahboninho"))
			return new Gahboninho();
		else if (name.equalsIgnoreCase("ValueModelAgent"))
			return new ValueModelAgent();
		else if (name.equalsIgnoreCase("Nice Tit-for-Tat Agent"))
			return a;// /agent doesn't work
		else if (name.equalsIgnoreCase("HardHeaded"))
			return new KLH();
		else if (name.equalsIgnoreCase("WinnerAgent"))
			return new WinnerAgent2();
		else if (name.equalsIgnoreCase("AgentK2"))
			return new Agent_K2();
		else if (name.equalsIgnoreCase("TheNegotiator"))
			return new TheNegotiator();
		else if (name.equalsIgnoreCase("IAMhaggler2011"))
			return a;// /agent doesn't work
		else if (name.equalsIgnoreCase("DNAgent"))
			return new DNAgent();
		else if (name.equalsIgnoreCase("BRAMAgent"))
			return new BRAMAgent();
		else if (name.equalsIgnoreCase("MrFriendly"))
			return new MrFriendly();
		else if (name.equalsIgnoreCase("ShAgent"))
			return new ShAgent();
		else if (name.equalsIgnoreCase("simpleAgent"))
			return new SimpleAgentNew();
		else
			return a;

	}

	@Override
	public Action chooseAction() {
		try {
			if (!agentInit) {
				if (actionOfPartner == null) // first bid, and Meta-Agent is the
												// first bidder (still not
												// initialized any agent)
				{
					first = true;
					return new Offer(this.getAgentID(),
							utilitySpace.getMaxUtilityBid());
				} else // second bid, or first bid as responder (and not
						// initiator)
				{
					agentInit = true;
					if (actionOfPartner instanceof Offer) {
						opponentFirstBid = utilitySpace
								.getUtility(((Offer) actionOfPartner).getBid());
					}

					// initialize agent.
					myAgent = selectAgent(getBestAgent());
					myAgent.internalInit(0, 1, this.startTime, this.totalTime,
							this.timeline, this.utilitySpace,
							this.parametervalues, getAgentID());
					myAgent.init();
					// send the message that received to the new agent..
					myAgent.ReceiveMessage(actionOfPartner);
				}
			}
			Action a = myAgent.chooseAction();
			if (a == null) {
				return new Offer(this.getAgentID(),
						utilitySpace.getMaxUtilityBid());
			} else { // handle reservation value, because all agent are from
						// Genius version 3.1 (and don't know what is
						// reservation value)
				double time = timeline.getTime();
				if (a instanceof Offer
						&& this.utilitySpace.getReservationValueWithDiscount(
								time) >= utilitySpace.getUtilityWithDiscount(
										((Offer) a).getBid(), time)) {
					a = new EndNegotiation(getAgentID());
				}

			}
			return a;

		} catch (Exception e) {
			return null;
		}
	}

	@Override
	public void ReceiveMessage(Action opponentAction) {
		actionOfPartner = opponentAction;
		if (agentInit)
			myAgent.ReceiveMessage(opponentAction);
	}

	@Override
	public String getName() {
		if (myAgentName == null || myAgentName == "")
			return "Meta-Agent 2012";
		else
			return "Meta-Agent 2012: " + myAgentName;
	}

	@Override
	public String getVersion() {
		return "1.42";
	}

	@Override
	public SupportedNegotiationSetting getSupportedNegotiationSetting() {
		return SupportedNegotiationSetting.getLinearUtilitySpaceInstance();
	}

	@Override
	public String getDescription() {
		return "ANAC2012";
	}
}
