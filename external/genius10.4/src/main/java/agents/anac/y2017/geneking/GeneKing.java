package agents.anac.y2017.geneking;

import java.util.List;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import java.io.IOException;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.EndNegotiation;
import genius.core.actions.Offer;
import genius.core.issue.ISSUETYPE;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.IssueInteger;
import genius.core.issue.IssueReal;
import genius.core.issue.Value;
import genius.core.issue.ValueDiscrete;
import genius.core.issue.ValueInteger;
import genius.core.issue.ValueReal;
import genius.core.list.Tuple;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;
import genius.core.persistent.StandardInfo;
import genius.core.persistent.StandardInfoList;
import genius.core.utility.EvaluatorDiscrete;

/**
 * @author W.Pasman Some improvements over the standard SimpleAgent.
 * 
 *         Random Walker, Zero Intelligence Agent
 */

class IssueFreqRank {// ある論点中の頻度等
	private HashMap<String, Integer> frequenceS;// 各提案内容の頻度
	private int sum;// 全頻度の合計
	private int maxFreq;// 最大頻度
	private double average;// 頻度の平均
	private double variance;// 頻度の分散
	private double pseudoWeight;// 疑似重み

	IssueFreqRank(List<ValueDiscrete> valueS, ValueDiscrete aSelected) {
		frequenceS = new HashMap<String, Integer>();
		for (ValueDiscrete valueD : valueS) {
			frequenceS.put(valueD.getValue(), 0);
		}
		sum = 0;
		maxFreq = 0;
		average = 0;
		variance = 0;
		pseudoWeight = 0;
		addFreq(aSelected.getValue());
	}

	public HashMap<String, Integer> getFrequenceS() {
		return (frequenceS);
	}

	public int getSum() {
		return (sum);
	}

	public int getMaxFreq() {
		return (maxFreq);
	}

	public double getAverage() {
		return (average);
	}

	public double getVariance() {
		return (variance);
	}

	public double pseudoWeight() {
		return (pseudoWeight);
	}

	public void addFreq(String aSelected) {
		int current = frequenceS.get(aSelected);
		frequenceS.put(aSelected, current + 1);
		sum += 1;
		average = sum / frequenceS.size();
		double vSum = 0;
		for (HashMap.Entry<String, Integer> entry : frequenceS.entrySet()) {// 全ての値を調べる
			if (maxFreq < entry.getValue()) {
				maxFreq = entry.getValue();// 最大頻度を更新
			}
			double diff = entry.getValue() - average;
			vSum += (diff * diff);
		}
		variance = Math.sqrt(vSum);
	}

}

public class GeneKing extends AbstractNegotiationParty {// Agent
	private Bid lastPartnerBid;
	private static double MINIMUM_BID_UTILITY = 0.3;
	//
	private StandardInfoList history;
	// private AbstractUtilitySpace utilitySpace;
	ArrayList<Bid> geneS = new ArrayList<Bid>();// 遺伝子(入札内容)
	ArrayList<Offer> foreignS = new ArrayList<Offer>();// 外来遺伝子(相手からの入札内容)
	ArrayList<Offer> tailForeignS = new ArrayList<Offer>();
	ArrayList<Bid> eliteS = new ArrayList<Bid>();// 優秀な個体(効用値の高い入札)
	Bid offeredBid = null;
	Bid myLastBid = null;
	long initNum = 2000;// 初期個体数
	int crossNum = 5;// 交叉回数
	int limitForeignSNum = 500;
	int maxForeignSNum = 200;
	int maxNum = 300;
	int MaxChangeNum = 100;
	double acceptableUtility = 0.0;
	double tgtUtility = 1.0;
	final double VALIANCE = 0.05;
	Random randomnr = new Random();
	EvaluatorDiscrete evaluator = new EvaluatorDiscrete();
	// HashMap<String,ArrayList<ArrayList<String>>> discreteRankS = new
	// HashMap<String,ArrayList<ArrayList<String>>>();
	HashMap<String, HashMap<String, Double>> dValueRankS = new HashMap<String, HashMap<String, Double>>();
	HashMap<AgentID, HashMap<String, IssueFreqRank>> dFreqRankS2 = new HashMap<AgentID, HashMap<String, IssueFreqRank>>();
	double utilWaight = 3.0, simWaight = 1.0;

	double getSimilarity2(Bid aBid1, Bid aBid2) {
		double sim = 0;
		List<Issue> IssueS1 = aBid1.getIssues();
		int index = 0;
		int len = IssueS1.size();
		while (index < len) {
			Issue issue1 = IssueS1.get(index);
			switch (issue1.getType()) {
			case DISCRETE: {
				ValueDiscrete valueDiscrete1 = (ValueDiscrete) aBid1
						.getValue(issue1.getNumber());
				ValueDiscrete valueDiscrete2 = (ValueDiscrete) aBid2
						.getValue(issue1.getNumber());
				HashMap<String, Double> drank = dValueRankS
						.get(issue1.getName());
				String fName = valueDiscrete1.getValue(),
						mName = valueDiscrete2.getValue();
				double bigger = drank.get(fName), smaller = drank.get(mName);
				if (bigger < smaller) {
					double temp = bigger;
					bigger = smaller;
					smaller = temp;
				}
				double diff = bigger - smaller;
				if (diff < 0) {
					diff *= -1;
				}
				double finalDiff = (1 - diff / drank.size()) / len;
				if (finalDiff < 0) {
					finalDiff = 0;
				}
				// System.out.println("fr="+bigger+" fn="+fName+" mr="+smaller+"
				// mn="+mName+" diff="+diff+" size="+drank.size());
				sim += finalDiff;
				break;
			}
			case INTEGER: {
				ValueInteger valueInt1 = (ValueInteger) aBid1
						.getValue(issue1.getNumber());
				ValueInteger valueInt2 = (ValueInteger) aBid2
						.getValue(issue1.getNumber());
				double diff = (double) valueInt1.getValue()
						- (double) valueInt2.getValue();
				// System.out.println("Idiff="+diff);
				if (diff < 0) {
					diff *= -1.0;
				}
				IssueInteger issueInt = (IssueInteger) IssueS1.get(index);
				double finalDiff = (1 - diff
						/ (issueInt.getUpperBound() - issueInt.getLowerBound()))
						/ len;
				if (finalDiff < 0) {
					finalDiff = 0;
				}
				sim += finalDiff;
				break;
			}
			default: {
				break;
			}
			}
			++index;
		}
		return (sim);
	}

	double getAveSim(Bid aAlpha, List<Offer> aBetaS) {
		double sim = 0;
		for (Offer beta : aBetaS) {
			sim += getSimilarity2(aAlpha, beta.getBid());
			// System.out.print(" sim="+sim);
		}
		sim /= aBetaS.size();
		// System.out.println(" total="+sim);
		return (sim);
	}

	double getExUtility2(Bid aBid, HashMap<String, IssueFreqRank> aFreqRankS) {
		double util = 0;
		List<Issue> issueS = utilitySpace.getDomain().getIssues();
		for (Issue issue : issueS) {
			switch (issue.getType()) {
			case DISCRETE: {
				IssueFreqRank freqRank = aFreqRankS.get(issue.getName());// 順位
				int freqValue = freqRank.getFrequenceS()
						.get(aBid.getValue(issue.getNumber()));
				if (freqValue == 0) {
					freqValue += 1;
				}
				double nFreqValue = (double) freqValue
						/ (double) freqRank.getMaxFreq();
				util += nFreqValue / issueS.size();
				break;
			}
			case INTEGER: {
				util += ((ValueInteger) (aBid.getValue(issue.getNumber())))
						.getValue();
				break;
			}
			}
		}
		return (util);
	}

	double getMyEvaluation(Bid aBid, List<Offer> aMotherS) {
		Set<AgentID> set = dFreqRankS2.keySet();
		double exUtil1 = 0, exUtil2 = 0, count = 0;
		for (AgentID id : set) {
			if (count == 0) {
				// exUtil1 = getExUtility(aBid,dFreqRankS.get(id));
				exUtil1 = getExUtility2(aBid, dFreqRankS2.get(id));
			} else {
				// exUtil2 = getExUtility(aBid,dFreqRankS.get(id));
				exUtil2 = getExUtility2(aBid, dFreqRankS2.get(id));
			}
			++count;
		}
		double diff = (exUtil1 - exUtil2) * 2;
		if (diff < 0) {
			diff *= -1;
		}
		double util = utilitySpace.getUtility(aBid),
				sim = getAveSim(aBid, aMotherS);
		double newEval = util * utilWaight + sim * simWaight + exUtil1 + exUtil2
				- diff;
		// System.out.println("(u,s,e1,e2,d)="+"("+util+","+sim+","+exUtil1+","+exUtil2+","+diff+")");
		return (newEval);
	}

	ArrayList<Bid> chooseEliteS3(ArrayList<Bid> aGeneS, List<Offer> aMotherS) {
		eliteS = new ArrayList<Bid>();
		Bid best = new Bid(aGeneS.get(0));
		double util = utilitySpace.getUtility(best);
		double sim = getAveSim(best, aMotherS);
		double bestEval = 0;
		// System.out.println(" (u,s)=("+util+","+sim+")");
		ArrayList<Bid> newGeneS = new ArrayList<Bid>(aGeneS);
		int index, count;
		while (eliteS.size() < maxNum) {
			index = 0;
			count = 0;
			bestEval = -10;
			for (Bid gene : newGeneS) {
				double eval = getMyEvaluation(gene, aMotherS);
				if (eval > bestEval) {
					best = gene;
					bestEval = eval;
					index = count;
				}
				++count;
			}
			eliteS.add(best);
			newGeneS.remove(index);// 加えた入札を消す
			if (newGeneS.size() == 0) {
				break;
			}
		}
		util = utilitySpace.getUtility(best);
		sim = getAveSim(best, aMotherS);
		// System.out.println(" (u,s)2=("+util+","+sim+")");
		return (eliteS);
	}

	Bid chooseBest(ArrayList<Bid> aGeneS) {
		Bid best = new Bid(aGeneS.get(0));
		double bestEval = getMyEvaluation(best, tailForeignS);
		for (Bid gene : aGeneS) {
			double eval = getMyEvaluation(gene, tailForeignS);
			if (eval > bestEval) {
				best = gene;
				bestEval = eval;
			}
		}
		return (best);
	}

	private Bid getRandomBidGK() {
		HashMap<Integer, Value> values = new HashMap<Integer, Value>(); // pairs
																		// <issuenumber,chosen
																		// value
																		// string>
		List<Issue> issues = utilitySpace.getDomain().getIssues();
		Random randomnr = new Random();
		Bid bid = null;
		do {
			for (Issue lIssue : issues) {
				switch (lIssue.getType()) {
				case DISCRETE: {// 文字列
					IssueDiscrete lIssueDiscrete = (IssueDiscrete) lIssue;
					int optionIndex = randomnr
							.nextInt(lIssueDiscrete.getNumberOfValues());
					values.put(lIssue.getNumber(),
							lIssueDiscrete.getValue(optionIndex));
					break;
				}
				case REAL: {//
					IssueReal lIssueReal = (IssueReal) lIssue;
					int optionInd = randomnr.nextInt(
							lIssueReal.getNumberOfDiscretizationSteps() - 1);
					values.put(lIssueReal.getNumber(), new ValueReal(lIssueReal
							.getLowerBound()
							+ (lIssueReal.getUpperBound()
									- lIssueReal.getLowerBound()) * (optionInd)
									/ (lIssueReal
											.getNumberOfDiscretizationSteps())));
					break;
				}
				case INTEGER: {
					IssueInteger lIssueInteger = (IssueInteger) lIssue;
					int optionIndex2 = lIssueInteger.getLowerBound()
							+ randomnr.nextInt(lIssueInteger.getUpperBound()
									- lIssueInteger.getLowerBound());
					values.put(lIssueInteger.getNumber(),
							new ValueInteger(optionIndex2));
					break;
				}
				default: {
					System.out.println("issue type " + lIssue.getType()
							+ " not supported by geneKing");
				}
				}
			}
			bid = new Bid(utilitySpace.getDomain(), values);
		} while (utilitySpace.getUtility(bid) < MINIMUM_BID_UTILITY);

		return (bid);
	}

	private ArrayList<Bid> getRandomBidS() {
		ArrayList<Bid> initS = new ArrayList<Bid>();
		while (initS.size() < initNum) {
			Bid newBid = null;
			// do{
			newBid = getRandomBidGK();
			// }while(initS.contains(newBid));
			initS.add(newBid);
		}
		return (initS);
	}

	Bid uniformCrossOver5(Bid aFather, Bid aMother,
			HashMap<String, IssueFreqRank> aFreqRankS) {
		Bid child = new Bid(aFather);
		List<Issue> genome = child.getIssues();
		for (Issue value : genome) {
			switch (value.getType()) {
			case DISCRETE: {
				IssueDiscrete lIssueDis = (IssueDiscrete) value;
				String fName = ((ValueDiscrete) aFather
						.getValue(value.getNumber())).getValue();// 父親の文字列
				String mName = ((ValueDiscrete) aMother
						.getValue(value.getNumber())).getValue();// 母親の文字列
				// System.out.println("f="+evaluator.getValue(((ValueDiscrete)aFather.getValue(value.getNumber())))
				// +"m="
				// +evaluator.getValue(((ValueDiscrete)aMother.getValue(value.getNumber()))));
				if (fName == mName) {// 互いに同じ提案だった場合
					child.putValue(value.getNumber(),
							aFather.getValue(value.getNumber()));
				} else {// 提案が異なる場合
					HashMap<String, Double> rank = dValueRankS
							.get(lIssueDis.getName());
					HashMap<String, Integer> aFreqRank = aFreqRankS
							.get(value.getName()).getFrequenceS();
					double bigger = rank.get(fName) * aFreqRank.get(fName),
							smaller = rank.get(mName) * aFreqRank.get(mName);
					if (bigger < smaller) {
						double temp = bigger;
						bigger = smaller;
						smaller = temp;
					}
					ArrayList<String> midS = new ArrayList<String>();
					for (HashMap.Entry<String, Double> entry : rank
							.entrySet()) {// 全ての値を調べる
						double newMid = entry.getValue()
								* aFreqRank.get(entry.getKey());
						if (bigger >= newMid && smaller <= newMid) {
							midS.add(entry.getKey());
						}
					}
					int newSeed = randomnr.nextInt(midS.size());
					child.putValue(value.getNumber(),
							new ValueDiscrete(midS.get(newSeed)));
				} // else
				break;
			}
			case INTEGER: {
				IssueInteger lIssueInteger = (IssueInteger) value;
				int bigger = ((ValueInteger) aFather
						.getValue(value.getNumber())).getValue();
				int smaller = ((ValueInteger) aMother
						.getValue(value.getNumber())).getValue();
				if (bigger < smaller) {
					int temp = bigger;
					bigger = smaller;
					smaller = temp;
				}
				int optionIndex = smaller
						+ randomnr.nextInt(bigger - smaller + 1);
				child.putValue(lIssueInteger.getNumber(),
						new ValueInteger(optionIndex));
				break;
			}
			default: {
				System.out.println("issue type " + value.getType()
						+ " not supported by geneKing");
			}
			}// switch
		} // for
		return (child);
	}

	// 突然変異を起こす
	Bid mutation(Bid aOriginal) {
		Bid mutant = new Bid(aOriginal);
		List<Issue> genome = mutant.getIssues();
		for (Issue issue : genome) {
			int rot = randomnr.nextInt(70);
			if (rot == 0) {// 値が一致したときのみ変異させる
				switch (issue.getType()) {
				case DISCRETE: {
					IssueDiscrete lIssueDiscrete = (IssueDiscrete) issue;
					int optionIndex = randomnr
							.nextInt(lIssueDiscrete.getNumberOfValues());
					mutant.putValue(issue.getNumber(),
							lIssueDiscrete.getValue(optionIndex));
					break;
				}
				case INTEGER: {
					IssueInteger lIssueInteger = (IssueInteger) issue;
					int optionIndex = lIssueInteger.getLowerBound()
							+ randomnr.nextInt(lIssueInteger.getUpperBound()
									- lIssueInteger.getLowerBound());
					mutant.putValue(lIssueInteger.getNumber(),
							new ValueInteger(optionIndex));
					break;
				}
				case REAL: {
					IssueReal lIssueReal = (IssueReal) issue;
					int optionInd = randomnr.nextInt(
							lIssueReal.getNumberOfDiscretizationSteps() - 1);
					mutant.putValue(lIssueReal.getNumber(), new ValueReal(
							lIssueReal.getLowerBound() + (lIssueReal
									.getUpperBound()
									- lIssueReal.getLowerBound()) * (optionInd)
									/ (lIssueReal
											.getNumberOfDiscretizationSteps())));
					break;
				}
				default: {
					System.out.println("issue type " + issue.getType()
							+ " not supported by geneKing");
				}
				}// switch
			} // if
		} // for
		return (mutant);
	}

	ArrayList<Bid> makeNextG3(ArrayList<Bid> aCurrentS,
			ArrayList<Offer> aForeignS) {
		ArrayList<Bid> nextGeneS = new ArrayList<Bid>();
		int tail = aForeignS.size();
		int len = 8;
		int start = tail - len;
		if (start < 0) {
			start = 0;
		}
		tailForeignS = new ArrayList<Offer>(aForeignS.subList(start, tail));
		// System.out.println("0Len="+aCurrentS.size()+"
		// 1Len="+aForeignS.size()+" 2Len="+tailForeignS.size()+"
		// start="+start+" tail="+tail);
		Bid lastBid = null;
		for (Bid alpha : aCurrentS) {
			for (Offer beta : tailForeignS) {
				for (int i = 0; i < crossNum; ++i) {
					// Bid child = uniformCrossOver3(alpha,beta.getBid());
					Bid child = uniformCrossOver5(alpha, beta.getBid(),
							dFreqRankS2.get(beta.getAgent()));
					child = mutation(child);
					double util = utilitySpace.getUtility(child);
					// System.out.print(util+ "");
					if (util > 0 && !child.equals(lastBid)) {
						nextGeneS.add(child);
						lastBid = child;
					}
				}
			}
		}
		// System.out.println("NextG="+nextGeneS.size());
		nextGeneS = chooseEliteS3(nextGeneS, tailForeignS);
		// System.out.println("NextG2="+nextGeneS.size());
		return (nextGeneS);
	}

	void genelogy() {
		if (foreignS.size() == 0) {
			Bid newBid = chooseBest(geneS);
			myLastBid = newBid;
			System.out.println("No foreignS!");
			return;
		} else {
			ArrayList<Bid> nextS = makeNextG3(geneS, foreignS);
			// System.out.println("NextLen="+nextS.size());
			Bid newBid = chooseBest(nextS);
			if (myLastBid != null) {
				// double newEval =
				// utilitySpace.getUtility(newBid)*utilWaight+getAveSim(newBid,foreignS)*simWaight;
				double newEval = getMyEvaluation(newBid, tailForeignS);
				double lastEval = getMyEvaluation(myLastBid, tailForeignS);
				if (newEval > lastEval) {
					myLastBid = newBid;
					// System.out.println("New Best!");
				}
			} else {
				myLastBid = newBid;
			}
			geneS = new ArrayList<Bid>(nextS);// 全入れ替え
		}
	}

	private boolean isAcceptableGK() {
		double offeredUtil = 0.0;
		if (offeredBid != null) {
			offeredUtil = utilitySpace.getUtility(offeredBid);
		}
		double myUtil = 1.0;
		if (myLastBid != null) {
			myUtil = utilitySpace.getUtility(myLastBid);
		}
		if (offeredUtil >= myUtil - VALIANCE) {// 自分の提案以上のものか、その近くであった場合
			return (true);
		}
		return (false);
	}

	private HashMap<String, Double> makeRank2(IssueDiscrete aIssue) {
		HashMap<String, Double> rRank = new HashMap<String, Double>();
		List<ValueDiscrete> valueS = aIssue.getValues();
		for (ValueDiscrete valueD : valueS) {
			// Integer newEval = evaluator.getValue(valueD);//評価値に直す
			double newEval = (double) (evaluator.getValue(valueD))
					/ evaluator.getEvalMax();
			System.out.println("newEval=" + newEval);
			// Integer newEval =
			// EvaluatorDiscrete.this.getEvaluationNotNormalized(valueD);
			rRank.put(valueD.getValue(), newEval);
		}
		return (rRank);
	}

	private void makeRankS(List<Issue> aIssueS) {
		System.out.println("Make Rank Start!");
		int count = 0;
		for (Issue issue : aIssueS) {
			if (issue.getType().equals(ISSUETYPE.DISCRETE)) {
				try {
					evaluator.loadFromXML(
							utilitySpace.toXML().getChildElementsAsList().get(0)
									.getChildElementsAsList().get(count));
					// evaluator.loadFromXML(issue.toXML());
					// evaluator.setXML(issue.toXML());
					// System.out.println("evaluator="+((EvaluatorDiscrete)evaluator).getEvalMax());
				} catch (IOException e) {

				}
				// discreteRankS.put(issue.getName(),
				// makeRank((IssueDiscrete)issue));
				dValueRankS.put(issue.getName(),
						makeRank2((IssueDiscrete) issue));
				System.out.println("issue-N Finish!");
			} // if
			++count;
		} // for
	}

	private void initInitNum() {
		long candidateNum = 1;
		candidateNum = utilitySpace.getDomain().getNumberOfPossibleBids();
		if (initNum > candidateNum) {
			initNum = candidateNum;
		}
	}

	private void initGK() {
		history = (StandardInfoList) getData().get();

		if (!history.isEmpty()) {
			// example of using the history. Compute for each party the maximum
			// utility of the bids in last session.
			Map<String, Double> maxutils = new HashMap<String, Double>();
			StandardInfo lastinfo = history.get(history.size() - 1);
			for (Tuple<String, Double> offered : lastinfo.getUtilities()) {
				String party = offered.get1();
				Double util = offered.get2();
				maxutils.put(party, maxutils.containsKey(party)
						? Math.max(maxutils.get(party), util) : util);
			}
			System.out.println(maxutils); // notice tournament suppresses all
											// output.
		}

		MINIMUM_BID_UTILITY = utilitySpace.getReservationValueUndiscounted();
		if (MINIMUM_BID_UTILITY < utilitySpace
				.getReservationValueUndiscounted()) {
			MINIMUM_BID_UTILITY = utilitySpace
					.getReservationValueUndiscounted();
		}
		acceptableUtility = MINIMUM_BID_UTILITY;
		System.out.println("get issueS!");
		List<Issue> issueS = utilitySpace.getDomain().getIssues();
		System.out.println("got issueS!");
		makeRankS(issueS);
		System.out.println("Made Rank");
		initInitNum();
		geneS = getRandomBidS();
		double bestUtil = 0;
		for (StandardInfo stdInfo : history) {
			Tuple agreement = stdInfo.getAgreement();
			if (agreement != null) {
				double newUtil = (double) agreement.get2();
				if (newUtil > bestUtil && newUtil > 0.7) {
					bestUtil = newUtil;
					myLastBid = (Bid) agreement.get1();
				}
				if (newUtil > 0.7) {
					geneS.add((Bid) agreement.get1());
				}
			}
			// System.out.println("profile="+stdInfo.getAgentProfiles());
			// System.out.println("stdInfo="+stdInfo.getUtilities());
			// System.out.println("agree="+stdInfo.getAgreement());
		}
	}

	private String getNameGK() {
		return ("GeneKing");
	}

	private double getAverageUtility() {
		double average = 0;
		for (Offer foreign : tailForeignS) {
			average += utilitySpace.getUtility(foreign.getBid());
		}
		average /= tailForeignS.size();
		return (average);
	}

	private Action chooseActionGK() {
		double time = timeline.getTime();
		if (time > 0.99) {// 締め切り間近の場合
			double partnerUtil = getAverageUtility();
			if (partnerUtil >= acceptableUtility) {
				acceptableUtility = partnerUtil;
			}

			if (getAverageUtility() < MINIMUM_BID_UTILITY) {
				System.out.println("End Negotiation!");
				return (new EndNegotiation(getPartyId()));
			} else if (partnerUtil >= acceptableUtility) {// 最後の提案が良かった場合
				return (new Accept(getPartyId(), lastPartnerBid));
			} else {
				genelogy();
				return (new Offer(getPartyId(), myLastBid));
			}
		} else {// まだ余裕のある場合
			if (isAcceptableGK()) {
				return (new Accept(getPartyId(), lastPartnerBid));
			} else {
				genelogy();
				return (new Offer(getPartyId(), myLastBid));
			}
		}
	}

	private IssueFreqRank makeFreqRank2(IssueDiscrete aIssue,
			ValueDiscrete aSelected) {
		IssueFreqRank rRank = new IssueFreqRank(aIssue.getValues(), aSelected);
		return (rRank);
	}

	private HashMap<String, IssueFreqRank> makeFreqRankS2(Bid aBid) {
		HashMap<String, IssueFreqRank> rFreqRankS = new HashMap<String, IssueFreqRank>();
		List<Issue> issueS = utilitySpace.getDomain().getIssues();
		for (Issue issue : issueS) {
			if (issue.getType().equals(ISSUETYPE.DISCRETE)) {// DISCRETE型だった場合
				IssueDiscrete dIssue = (IssueDiscrete) issue;
				rFreqRankS.put(issue.getName(), makeFreqRank2(dIssue,
						(ValueDiscrete) (aBid.getValue(dIssue.getNumber()))));
			}
		}
		return (rFreqRankS);
	}

	private void updateFreqRank2(ValueDiscrete aSelected, IssueFreqRank aRank) {
		aRank.addFreq(aSelected.getValue());
	}

	void updateFreqRankS2(Bid aBid,
			HashMap<String, IssueFreqRank> adFreqRankS) {
		List<Issue> issueS = utilitySpace.getDomain().getIssues();
		for (Issue issue : issueS) {
			if (issue.getType().equals(ISSUETYPE.DISCRETE)) {// DISCRETE型だった場合
				updateFreqRank2(
						(ValueDiscrete) (aBid.getValue(issue.getNumber())),
						adFreqRankS.get(issue.getName()));
			}
		}
	}

	// Offerのリストの長さをチェックし、長すぎた場合は後方のみを残す
	ArrayList<Offer> reduceList(ArrayList<Offer> aList, int aMaxSize) {
		ArrayList<Offer> rList = new ArrayList<Offer>(
				aList.subList(aList.size() - aMaxSize, aList.size()));
		return (rList);
	}

	private void ReceiveMessageGK(Action aAction) {
		AgentID id = aAction.getAgent();
		if (aAction instanceof Offer) {
			offeredBid = new Bid(((Offer) aAction).getBid()); // 提案された合意案候補
			foreignS.add((Offer) aAction);
			if (dFreqRankS2.containsKey(id)) {// 登録されていた場合
				updateFreqRankS2(offeredBid, dFreqRankS2.get(id));
			} else {// まだ登録されていない場合
				dFreqRankS2.put(id, makeFreqRankS2(offeredBid));
			}
			lastPartnerBid = offeredBid;
			// System.out.println("aAction="+offeredBid+"
			// len="+foreignS.size()+" "+aAction.getAgent());
		} else if (aAction instanceof Accept) {// 受け入れだった場合
			Bid accepted = new Bid(((Accept) aAction).getBid());
			Offer accOffer = new Offer(aAction.getAgent(), accepted);
			foreignS.add(accOffer);
			if (dFreqRankS2.containsKey(id)) {// 登録されていた場合
				updateFreqRankS2(accepted, dFreqRankS2.get(id));
			} else {// まだ登録されていない場合
				dFreqRankS2.put(id, makeFreqRankS2(accepted));
			}
		}
		if (foreignS.size() > limitForeignSNum) {
			foreignS = reduceList(foreignS, maxForeignSNum);// 数が増えすぎたら昔のものを減らす
		}

	}

	@Override
	public void init(NegotiationInfo info) {
		super.init(info);
		System.out.println("Discount Factor is "
				+ getUtilitySpace().getDiscountFactor());
		System.out.println("Reservation Value is "
				+ getUtilitySpace().getReservationValueUndiscounted());
		MINIMUM_BID_UTILITY = getUtilitySpace()
				.getReservationValueUndiscounted();
		initGK();
	}

	@Override
	public Action chooseAction(List<Class<? extends Action>> validActions) {
		return (chooseActionGK());
	}

	@Override
	public void receiveMessage(AgentID sender, Action action) {
		super.receiveMessage(sender, action);
		ReceiveMessageGK(action);
	}

	@Override
	public String getDescription() {
		return "ANAC2017";
	}
}
