package agents.anac.y2016.maxoops;

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import agents.Jama.Matrix;

import java.util.Random;
import java.util.Set;

import genius.core.Bid;
import genius.core.BidHistory;
import genius.core.Domain;
import genius.core.bidding.BidDetails;
import genius.core.issue.Objective;
import genius.core.issue.Value;
import genius.core.issue.ValueDiscrete;
import genius.core.issue.ValueInteger;
import genius.core.issue.ValueReal;
import genius.core.utility.AdditiveUtilitySpace;
import genius.core.utility.EVALUATORTYPE;
import genius.core.utility.Evaluator;
import genius.core.utility.EvaluatorDiscrete;
import genius.core.utility.EvaluatorInteger;
import genius.core.utility.EvaluatorReal;

public class OPTComponent {

	// Adjustable Parameters
	protected final double flrate;

	MaxOops agent;
	Domain domain;

	public AdditiveUtilitySpace myUtilitySpace;
	public HashMap<Objective, Evaluator> fEvaluators;
	public Payoff[] payoffs;
	public Matrix w;

	public OPTComponent(MaxOops agent, AdditiveUtilitySpace utilitySpace) {
		// Set Adjustable Parameters First
		agent.params.addParam("OPTComponent.flrate", 0.1);
		this.flrate = agent.params.getParam("OPTComponent.flrate");
		this.agent = agent;
		this.myUtilitySpace = utilitySpace;
		this.domain = utilitySpace.getDomain();
		this.fEvaluators = new HashMap<Objective, Evaluator>();
		Set<Entry<Objective, Evaluator>> fEvaluatorsSet = this.myUtilitySpace
				.getEvaluators();
		for (Entry<Objective, Evaluator> entry : fEvaluatorsSet) {
			this.fEvaluators.put(entry.getKey(), entry.getValue());
		}
		this.w = new Matrix(agent.numIssues, 1);
		Objective root = agent.domain.getObjectivesRoot();
		int i = 0;
		for (Enumeration<Objective> issueEnum = root
				.getPreorderIssueEnumeration(); issueEnum.hasMoreElements();) {
			Objective is = (Objective) issueEnum.nextElement();
			Evaluator eval = (Evaluator) fEvaluators.get(is);
			this.w.set(i++, 0, eval.getWeight());
		}
		this.payoffs = new Payoff[agent.numParties - 1];
		for (i = 0; i < agent.numParties - 1; i++) {
			this.payoffs[i] = new Payoff(i, agent, myUtilitySpace, this);
		}
	}

	public Bid getOptimalBidByThredhold(double f_thre) {
		Bid[] firstBid = new Bid[agent.numParties - 1];
		Bid[] bestBid = new Bid[agent.numParties - 1];
		Bid[] lastBid = new Bid[agent.numParties - 1];
		for (int i = 0; i < payoffs.length; i++) {
			if (!agent.opponentsDistinctBids[i].isEmpty()) {
				firstBid[i] = agent.opponentsDistinctBids[i]
						.getFirstBidDetails().getBid();
				bestBid[i] = agent.opponentsDistinctBids[i].getBestBidDetails()
						.getBid();
				lastBid[i] = agent.lastBid;
			}
		}
		BidHistory bidsBySumOfPayoffs = new BidHistory();
		int lbInd = Math.max(0, (int) (f_thre * 100) - 1);
		int ubInd = Math.min((int) ((f_thre + 1. * agent.stdUtil) * 100) + 1,
				100);
		HashSet<Bid> searchBids = new HashSet<Bid>();
		for (int i = lbInd; i <= ubInd; i++) {
			searchBids.addAll(agent.hashBids.get(i));
		}
		if (agent.DMC.timeline.getTime() > 0.99 * agent.delta
				&& searchBids.size() < 1) {
			for (int i = ubInd + 1; i <= 100; i++) {
				searchBids.addAll(agent.hashBids.get(i));
			}
		}
		MaxOops.log3.println("ubInd = " + ubInd + "lbInd = " + lbInd
				+ "f_thre = " + f_thre + ", bids size =" + searchBids.size());
		Iterator<Bid> bidIter = searchBids.iterator();
		Random rand = new Random((long) (f_thre * 10000) % 1001);
		while (bidIter.hasNext()) {
			Bid bid = bidIter.next();
			double sumOfPayoffs = 0.;
			for (int i = 0; i < agent.numParties; i++) {
				if (i == agent.numParties - 1) {
					sumOfPayoffs += myUtilitySpace.getUtility(bid);
					break;
				}
				double sim1 = payoffs[i].getSimilarity(firstBid[i], bid) * 0.3
						* rand.nextFloat();
				double sim2 = payoffs[i].getSimilarity(bestBid[i], bid) * 0.2
						* rand.nextFloat();
				double sim3 = payoffs[i].getSimilarity(lastBid[i], bid) * 0.1
						* rand.nextFloat();
				sumOfPayoffs += payoffs[i].getPayoff(bid)
						+ (sim1 + sim2 + sim3) * rand.nextFloat()
						* agent.stdUtil * 10;
			}
			bidsBySumOfPayoffs.add(new BidDetails(bid, sumOfPayoffs,
					myUtilitySpace.getUtility(bid)));
		}
		if (bidsBySumOfPayoffs.isEmpty()) {
			return null;
		}
		return bidsBySumOfPayoffs.getBestBidDetails().getBid();
	}

	public Matrix normalise(Matrix x) {
		double min = 0, max = 0;
		double[][] arr = x.getArray();
		for (double[] vec : arr) {
			for (double val : vec) {
				min = Math.min(min, val);
				max = Math.max(max, val);
			}
		}
		x = x.plus(new Matrix(agent.numIssues, 1, Math.abs(min)));
		x = x.times(1. / x.norm1());
		return x;
	}

	public Matrix getMatrixFromBids(List<Bid> bids) {
		int rows = bids.size();
		Matrix X = new Matrix(rows, agent.numIssues);
		for (int i = 0; i < rows; i++) {
			X.setMatrix(new int[] { i }, 0, agent.numIssues - 1,
					getVectorFromBid(bids.get(i)).transpose());
		}
		return X;
	}

	public List<Value> getValuesFromBid(Bid bid) {
		Objective root = agent.domain.getObjectivesRoot();
		List<Value> values = new ArrayList<Value>();
		for (Enumeration<Objective> issueEnum = root
				.getPreorderIssueEnumeration(); issueEnum.hasMoreElements();) {
			Objective is = (Objective) issueEnum.nextElement();
			Evaluator eval = (Evaluator) fEvaluators.get(is);
			EVALUATORTYPE type = eval.getType();
			switch (type) {
			case REAL:
				try {
					values.add((ValueReal) bid.getValue(is.getNumber()));
				} catch (Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				break;
			case DISCRETE:
				try {
					values.add((ValueDiscrete) bid.getValue(is.getNumber()));
				} catch (Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				break;
			case INTEGER:
				values.add((ValueInteger) bid.getValue(is.getNumber()));
				break;
			default:
				break;
			}
		}
		return values;
	}

	public Matrix getVectorFromBid(Bid bid) {
		Objective root = agent.domain.getObjectivesRoot();
		Matrix x = new Matrix(agent.numIssues, 1);
		int i = 0;
		for (Enumeration<Objective> issueEnum = root
				.getPreorderIssueEnumeration(); issueEnum.hasMoreElements();) {
			Objective is = (Objective) issueEnum.nextElement();
			Evaluator eval = (Evaluator) fEvaluators.get(is);
			double xi = 0;
			EVALUATORTYPE type = eval.getType();
			switch (type) {
			case REAL:
				EvaluatorReal evalReal = (EvaluatorReal) eval;
				try {
					xi = evalReal.getEvaluation(((ValueReal) bid.getValue(is
							.getNumber())).getValue());
				} catch (Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				break;
			case DISCRETE:
				EvaluatorDiscrete evalDis = (EvaluatorDiscrete) eval;
				try {
					xi = evalDis.getEvaluation((ValueDiscrete) bid.getValue(is
							.getNumber()));
				} catch (Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				break;
			case INTEGER:
				EvaluatorInteger evalInt = (EvaluatorInteger) eval;
				xi = evalInt.getEvaluation(((ValueInteger) bid.getValue(is
						.getNumber())).getValue());
				break;
			default:
				xi = eval.getEvaluation(myUtilitySpace, bid, is.getNumber());
				break;
			}
			x.set(i++, 0, xi);
		}
		return x;
	}
}

class Payoff {

	protected final int updateIterationLimit;

	int opponent;
	MaxOops agent;
	Matrix w;
	AdditiveUtilitySpace myUtilitySpace;
	OPTComponent mother;
	List<Bid> minibatchBidList;
	int minibatchUpdateSize;
	List<Map<Value, Integer>> freq;
	List<Map<Value, Double>> eval;
	double lastAvgPayoff;
	private int iteration = 0;
	private Matrix mem = null, g = null, g2 = null;

	public Payoff(int opponent, MaxOops agent,
			AdditiveUtilitySpace myUtilitySpace, OPTComponent mother) {
		agent.params.addParam("OPTComponent.updateIterationLimit", 50);
		this.updateIterationLimit = (int) agent.params
				.getParam("OPTComponent.updateIterationLimit");
		this.opponent = opponent;
		this.iteration = 0;
		this.agent = agent;
		this.myUtilitySpace = myUtilitySpace;
		this.minibatchBidList = new ArrayList<Bid>();
		this.minibatchUpdateSize = 10;
		this.mother = mother;
		this.lastAvgPayoff = agent.secMaxUtil;
		this.w = new Matrix(agent.numIssues, 1, 1. / agent.numIssues);
		this.w = this.w.plus(mother.w).times(0.5);
		this.freq = new ArrayList<Map<Value, Integer>>();
		this.eval = new ArrayList<Map<Value, Double>>();
		Objective root = agent.domain.getObjectivesRoot();
		for (Enumeration<Objective> issueEnum = root
				.getPreorderIssueEnumeration(); issueEnum.hasMoreElements();) {
			Map<Value, Integer> valuesFreq = new HashMap<Value, Integer>();
			Map<Value, Double> valuesEval = new HashMap<Value, Double>();
			Objective is = (Objective) issueEnum.nextElement();
			Evaluator e = (Evaluator) mother.fEvaluators.get(is);
			EVALUATORTYPE type = e.getType();
			Double xi = 0.;
			switch (type) {
			case REAL:
				EvaluatorReal evalReal = (EvaluatorReal) e;
				Double range = evalReal.getUpperBound()
						- evalReal.getLowerBound();
				for (Integer vi = 0; vi < 500; vi++) {
					valuesFreq.put(new ValueInteger(vi), 0);
					try {
						Double v = range * vi / 500. + evalReal.getLowerBound();
						xi = evalReal.getEvaluation(v);
					} catch (Exception err) {
						// TODO Auto-generated catch block
						err.printStackTrace();
					}
					valuesEval.put(new ValueInteger(vi), xi);
				}
				break;
			case DISCRETE:
				EvaluatorDiscrete evalDis = (EvaluatorDiscrete) e;
				Set<ValueDiscrete> values = evalDis.getValues();
				for (ValueDiscrete v : values) {
					valuesFreq.put(v, 0);
					try {
						xi = evalDis.getEvaluation(v);
					} catch (Exception err) {
						// TODO Auto-generated catch block
						err.printStackTrace();
					}
					valuesEval.put(v, xi);
				}
				break;
			case INTEGER:
				EvaluatorInteger evalInt = (EvaluatorInteger) e;
				for (Integer v = evalInt.getLowerBound(); v <= evalInt
						.getUpperBound(); v++) {
					valuesFreq.put(new ValueInteger(v), 0);
					xi = evalInt.getEvaluation(v);
					valuesEval.put(new ValueInteger(v), xi);
				}
				break;
			default:
				break;
			}
			this.freq.add(valuesFreq);
			this.eval.add(valuesEval);
		}
	}

	public void updateFrequency(Bid bid) {
		List<Value> values = mother.getValuesFromBid(bid);
		Objective root = agent.domain.getObjectivesRoot();
		int i = 0;
		for (Enumeration<Objective> issueEnum = root
				.getPreorderIssueEnumeration(); issueEnum.hasMoreElements();) {
			Objective is = (Objective) issueEnum.nextElement();
			Evaluator e = (Evaluator) mother.fEvaluators.get(is);
			EVALUATORTYPE type = e.getType();
			switch (type) {
			case REAL:
				EvaluatorReal evalReal = (EvaluatorReal) e;
				Double v = ((ValueReal) values.get(i)).getValue();
				Double range = evalReal.getUpperBound()
						- evalReal.getLowerBound();
				Integer vi = (int) ((v - evalReal.getLowerBound()) * 500. / range);
				ValueInteger Vi = new ValueInteger(vi);
				this.freq.get(i).put(Vi, 1 + this.freq.get(i).get(Vi));
				break;
			case DISCRETE:
				if (!this.freq.get(i).containsKey(values.get(i))) {
					this.freq.get(i).put(values.get(i), 1);
				} else {
					this.freq.get(i).put(values.get(i),
							1 + this.freq.get(i).get(values.get(i)));
				}
				break;
			case INTEGER:
				if (!this.freq.get(i).containsKey(values.get(i))) {
					this.freq.get(i).put(values.get(i), 1);
				} else {
					this.freq.get(i).put(values.get(i),
							1 + this.freq.get(i).get(values.get(i)));
				}
				break;
			default:
				break;
			}
			i++;
		}
	}

	public void updateEvaluation() {
		if (iteration >= updateIterationLimit / 2) {
			return;
		}
		for (int i = 0; i < this.freq.size(); i++) {
			double time = agent.DMC.timeline.getTime();
			Double maxOfLogFreq = 0.;
			for (Integer f : this.freq.get(i).values()) {
				maxOfLogFreq = Math.max(Math.log(f + 1), maxOfLogFreq);
			}
			double wt = (0.5 - time) / 3.;
			for (Value e : this.eval.get(i).keySet()) {
				Integer f = this.freq.get(i).get(e);
				Double ori = this.eval.get(i).get(e);
				this.eval.get(i).put(e,
						ori * (1 - wt) + wt * (Math.log(f + 1) / maxOfLogFreq));
			}
			Double maxOfEvals = 0.;
			for (Double e : this.eval.get(i).values()) {
				maxOfEvals = Math.max(e, maxOfEvals);
			}
			for (Value e : this.eval.get(i).keySet()) {
				this.eval.get(i).put(e, this.eval.get(i).get(e) / maxOfEvals);
			}
		}
	}

	public void initWeights(Bid firstBid) {
		iteration = 1;
		this.updateFrequency(firstBid);
		Matrix tw = mother.normalise(mother.getVectorFromBid(firstBid));
		double min = 0, max = 0;
		double[][] arr = tw.getArray();
		for (double[] vec : arr) {
			for (double val : vec) {
				min = Math.min(min, val);
				max = Math.max(max, val);
			}
		}
		double updateWeight = 1. / (1. + (max - min));
		this.w = tw.times(updateWeight).plus(this.w.times(1 - updateWeight));
	}

	public void updateWeights(Bid newBid) {
		if (iteration >= updateIterationLimit
				&& agent.DMC.timeline.getTime() > 0.6) {
			return;
		}
		this.updateFrequency(newBid);
		minibatchBidList.add(newBid);
		if (minibatchBidList.size() < minibatchUpdateSize) {
			return;
		}
		this.updateEvaluation();
		iteration += 1;
		Matrix tw = w.copy();
		Matrix X = mother.getMatrixFromBids(minibatchBidList);
		Matrix y = new Matrix(minibatchUpdateSize, 1, lastAvgPayoff);
		Matrix y_ = X.times(w);
		Matrix dfdw = new Matrix(agent.numIssues, 1);
		dfdw = X.transpose().times(y_.minus(y)).times(2. / minibatchUpdateSize);
		lastAvgPayoff = y_.norm1() / minibatchUpdateSize;
		minibatchBidList.clear();
		Matrix ones = new Matrix(agent.numIssues, 1, 1.);
		Matrix zeros = new Matrix(agent.numIssues, 1);
		if (mem == null || g == null || g2 == null) {
			mem = ones.copy();
			g = zeros.copy();
			g2 = zeros.copy();
		}
		Matrix r = ones.arrayRightDivide(mem.plus(ones));
		g = g.arrayTimes(ones.minus(r)).plus(r.arrayTimes(dfdw));
		g2 = g2.arrayTimes(ones.minus(r)).plus(
				r.arrayTimes(dfdw.arrayTimes(dfdw)));
		mem = mem.arrayTimes(ones.times(1 - mother.flrate)).plus(ones);
		Matrix alrate = g.arrayTimes(g).arrayRightDivide(
				g2.plus(ones.times(1e-13)));
		for (int i = 0; i < agent.numIssues; i++) {
			double alrateij = Math.min(alrate.get(i, 0), mother.flrate);
			alrate.set(i, 0, alrateij / (Math.sqrt(g2.get(i, 0)) + 1e-13));
		}
		double avgPayoffDiff = dfdw.norm1();
		tw = mother.normalise(tw.minus(dfdw.arrayTimes(alrate))).times(
				avgPayoffDiff);
		tw = tw.plus(w.times(1. - avgPayoffDiff));
		double min = 0, max = 0;
		double[][] arr = tw.getArray();
		for (double[] vec : arr) {
			for (double val : vec) {
				min = Math.min(min, val);
				max = Math.max(max, val);
			}
		}
		this.w = tw;
	}

	public double getPayoff(Bid bid) {
		double payoff = 0;
		payoff = w.arrayTimes(getVectorFromBid(bid)).norm1();
		return payoff;
	}

	public double getSimilarity(Bid bid1, Bid bid2) {
		try {
			Matrix x1 = getVectorFromBid(bid1);
			Matrix x2 = getVectorFromBid(bid2);
			return (x1.arrayTimes(x2)).norm1() * 1. / x1.norm1() / x2.norm1();
		} catch (Exception err) {
			return 0;
		}
	}

	public Matrix getVectorFromBid(Bid bid) {
		Matrix x = new Matrix(agent.numIssues, 1);
		List<Value> values = mother.getValuesFromBid(bid);
		int i = 0;
		Objective root = agent.domain.getObjectivesRoot();
		for (Enumeration<Objective> issueEnum = root
				.getPreorderIssueEnumeration(); issueEnum.hasMoreElements();) {
			Objective is = (Objective) issueEnum.nextElement();
			Evaluator e = (Evaluator) mother.fEvaluators.get(is);
			EVALUATORTYPE type = e.getType();
			switch (type) {
			case REAL:
				EvaluatorReal evalReal = (EvaluatorReal) e;
				Double v = ((ValueReal) values.get(i)).getValue();
				Double range = evalReal.getUpperBound()
						- evalReal.getLowerBound();
				Integer vi = (int) ((v - evalReal.getLowerBound()) * 500. / range);
				ValueInteger Vi = new ValueInteger(vi);
				x.set(i, 0, this.eval.get(i).get(Vi));
				break;
			case DISCRETE:
				x.set(i, 0, this.eval.get(i).get(values.get(i)));
				break;
			case INTEGER:
				x.set(i, 0, this.eval.get(i).get(values.get(i)));
				break;
			default:
				break;
			}
			i++;
		}
		return x;
	}

}