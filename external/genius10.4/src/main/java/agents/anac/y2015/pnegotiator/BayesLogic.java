package agents.anac.y2015.pnegotiator;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.TreeSet;

import genius.core.Bid;
import genius.core.Domain;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.ValueDiscrete;
import genius.core.utility.AdditiveUtilitySpace;
import genius.core.utility.EvaluatorDiscrete;

/**
 * Created by chad on 2/28/15.
 */
public class BayesLogic {
	private AdditiveUtilitySpace utilitySpace;
	private ArrayList<TreeSet<ValueFrequency>> valueFrequencies;
	private ArrayList<ValueFrequency>[] vFreqs;
	public int T = 1;
	int totalBids = 0;
	int P;
	public int V = 1;

	public BayesLogic(AdditiveUtilitySpace utilitySpace, int P)
			throws Exception {
		Domain domain = utilitySpace.getDomain();
		List<Issue> issues = domain.getIssues();
		this.P = P;

		this.utilitySpace = utilitySpace;
		valueFrequencies = new ArrayList<TreeSet<ValueFrequency>>(
				issues.size() - 1);
		vFreqs = new ArrayList[issues.size() - 1];
		for (int i = 0; i < issues.size() - 1; ++i) {
			// System.out.println("Adding TreeSet<ValueFrequency> vfs[" + (i) +
			// "]");
			valueFrequencies.add(i, new TreeSet<ValueFrequency>(new VFComp()));
			vFreqs[i] = new ArrayList<ValueFrequency>();
			IssueDiscrete di = (IssueDiscrete) domain.getIssues().get(i);
			EvaluatorDiscrete ed = (EvaluatorDiscrete) utilitySpace
					.getEvaluator(i + 1);
			for (ValueDiscrete v : di.getValues()) {
				// System.out.println("\tAdding ValueFrequency element[" +
				// v.value + ", " + (ed.getEvaluation(v) * ed.getWeight()) +
				// "]");
				ValueFrequency v1 = new ValueFrequency(v, ed.getEvaluation(v),
						P);
				valueFrequencies.get(i).add(v1);
				vFreqs[i].add(v1);
			}
		}

	}

	public Bid bayesBid(Bid bestBid) {
		Bid b = new Bid(bestBid);
		for (int i = 0; i < valueFrequencies.size(); ++i) {
			Iterator<ValueFrequency> itr = valueFrequencies.get(i)
					.descendingIterator();
			ValueDiscrete highestExpectedUtilVal = null;
			double highestExpectedUtil = -1.0;
			// System.out.format("Issue: %d\n", i);
			while (itr.hasNext()) {
				ValueFrequency vfr = itr.next();
				double FP = 1;
				for (int p = 0; p < P; p++)
					FP *= vfr.opponentFrequency[p];
				// Math.pow(totalBids,P)
				double EU = FP * Math.pow(vfr.utility, V);
				// System.out.format("  Choice: %10s, U = %6.4f, E[U]: %10.2f,  Frequencies: %s\n",
				// vfr.value, vfr.utility, EU,
				// Arrays.toString(vfr.opponentFrequency));
				if (EU > highestExpectedUtil) {
					highestExpectedUtilVal = vfr.value;
					highestExpectedUtil = EU;
				}
			}
			// ValueDiscrete highestExpectedUtilVal =
			// (ValueDiscrete)(lastBid.getValue(i + 1));
			// while(itr.hasNext()) {
			// ValueFrequency vfr = itr.next();
			// if((((double)vfr.opponentFrequency)/(double)totalBids) *
			// vfr.utility >= highestExpectedUtil) {
			// highestExpectedUtil =
			// ((double)vfr.opponentFrequency/(double)totalBids * vfr.utility);
			// highestExpectedUtilVal = vfr.value;
			// }
			// }
			b = b.putValue(i + 1, highestExpectedUtilVal);
		}
		return b;
	}

	private Bid asBid(ValueFrequency[] b) throws Exception {
		Bid b1 = utilitySpace.getMaxUtilityBid();
		for (int i = 0; i < b.length; i++) {
			b1 = b1.putValue(i + 1, b[i].value);
		}
		return b1;
	}

	private double EU2(ValueFrequency[] b) throws Exception {
		double EU = utilitySpace.getUtility(asBid(b));
		for (int p = 1; p < P; p++) {
			for (int v = 0; v < b.length; v++)
				EU *= b[v].opponentFrequency[p];
		}
		return EU;
	}

	/**
	 * Approximates bid value
	 * 
	 * @return
	 */
	public Bid bayesBid2(Random rand) throws Exception {
		ValueFrequency[] b = new ValueFrequency[valueFrequencies.size()];
		for (int i = 0; i < valueFrequencies.size(); i++)
			b[i] = valueFrequencies.get(i).last();
		double T = 1, Tmin = 0.2, alpha = 0.9;
		int K = 10;
		double EU = EU2(b);
		do {
			System.out.println(T);
			for (int k = 0; k < K; k++) {
				ValueFrequency[] nb = Arrays.copyOf(b, b.length);

				int ri = rand.nextInt(nb.length);
				nb[ri] = vFreqs[ri].get(rand.nextInt(vFreqs[ri].size()));

				double nEU = EU2(nb);
				double p = Math.exp((EU - nEU) / T);
				if (p > rand.nextDouble()) {
					b = nb;
					EU = nEU;
				}
			}
			T *= alpha;
		} while (T > Tmin);
		return asBid(b);
	}

	// Update the frequency of our proposed values based on a bid we're about to
	// make
	public void updateOurFrequency(Bid bid) throws Exception {
		for (int i = 0; i < valueFrequencies.size(); ++i) {
			Iterator<ValueFrequency> itr = valueFrequencies.get(i)
					.descendingIterator();
			while (itr.hasNext()) {
				ValueFrequency vfr = itr.next();
				if (vfr.value.toString().compareTo(
						((ValueDiscrete) bid.getValue(i + 1)).toString()) == 0) {
					vfr.ourFrequency++;
				}
				break;
			}
		}
	}

	// Update the frequency of opponents' proposed values based on a bid an
	// opponent just made
	public void updateOpponentFrequency(Bid bid, int P) throws Exception {
		++totalBids;
		for (int i = 0; i < valueFrequencies.size(); ++i) {
			Iterator<ValueFrequency> itr = valueFrequencies.get(i)
					.descendingIterator();
			while (itr.hasNext()) {
				ValueFrequency vfr = itr.next();
				// if(vfr.value.toString().compareTo(((ValueDiscrete)bid.getValue(i+1)).toString())
				// == 0) {
				// System.out.println(vfr + " " + bid);
				if (vfr.value.toString().equals(bid.getValue(i + 1).toString())) {
					vfr.opponentFrequency[P] += T;
					break;
				}
			}
		}
	}
}
