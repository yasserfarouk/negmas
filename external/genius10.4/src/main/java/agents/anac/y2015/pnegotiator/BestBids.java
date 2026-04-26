package agents.anac.y2015.pnegotiator;

import java.util.Arrays;
import java.util.Random;

import genius.core.Bid;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.ValueDiscrete;
import genius.core.utility.AdditiveUtilitySpace;
import genius.core.utility.EvaluatorDiscrete;

/**
 * Created by chad on 2/27/15.
 */
public class BestBids {
	// private Bid minBid;
	// private ArrayList<Tuple<Bid,Double>> preferredBids;

	int[] iprefs;
	ValueDiscrete[][] vs;
	int[][] vprefs;

	AdditiveUtilitySpace utilitySpace;

	public BestBids(AdditiveUtilitySpace utilitySpace) {
		this.utilitySpace = utilitySpace;
		try {
			this.genIssuePreferenceOrder();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private void genIssuePreferenceOrder() throws Exception {
		int N = utilitySpace.getDomain().getIssues().size();
		Tuple<Integer, Double> issues[] = new Tuple[N];
		iprefs = new int[N];
		vprefs = new int[N][];
		vs = new ValueDiscrete[N][];
		for (int i = 0; i < N; i++) {
			IssueDiscrete di = (IssueDiscrete) utilitySpace.getDomain()
					.getIssues().get(i);
			EvaluatorDiscrete de = (EvaluatorDiscrete) utilitySpace
					.getEvaluator(i + 1);
			issues[i] = new Tuple<Integer, Double>(i, de.getWeight());
			int M = di.getNumberOfValues();
			Tuple<Integer, Double> values[] = new Tuple[M];
			vs[i] = new ValueDiscrete[M];
			for (int j = 0; j < M; j++) {
				double w = de.getEvaluation(di.getValue(j));
				values[j] = new Tuple<Integer, Double>(j, w);
				vs[i][j] = di.getValue(j);
			}
			Arrays.sort(values);
			vprefs[i] = new int[M];
			for (int k = 0; k < M; k++)
				vprefs[i][k] = values[k].key;
		}
		Arrays.sort(issues);
		for (int i = 0; i < N; i++)
			iprefs[i] = issues[i].key;
	}

	public int getValue(int issue, ValueDiscrete v) {
		for (int j = 0; j < vs[issue].length; j++) {
			if (vs[issue][j] == v)
				return j;
		}
		return -1;
	}

	private static <T> int indexOf(int[] arr, int item) {
		for (int i = 0; i < arr.length; i++) {
			if (arr[i] == item)
				return i;
		}
		return -1;
	}

	public Bid getRandomBid(Bid r, Random rand, double minValue, double maxValue)
			throws Exception {
		Bid b = new Bid(r);
		int N = iprefs.length;
		// int[] vs = new int[N];
		// for(int i = 0; i < N; i++)
		// vs[iprefs[i]] = getValue(i, (ValueDiscrete)b.getValue(i));
		double U = utilitySpace.getUtility(b);
		System.out.println(Arrays.toString(iprefs));
		int t = 0;
		while (U < minValue || U >= maxValue) {
			t++;
			if (t > 100)
				break;
			// System.out.println(U);
			int ri = rand.nextInt(N);// .get((int)((1.-rand.nextDouble()*rand.nextDouble())*(available.size()-1)));
			int rI = iprefs[ri];
			int v = getValue(rI, (ValueDiscrete) b.getValue(rI + 1));
			int V = indexOf(vprefs[rI], v);
			int M = vprefs[rI].length;
			if (U < minValue) { // increase the preference of an issue to the
								// next highest level
				if (V >= M - 1) {
					continue;
				}
				// System.out.format("%d %d %d", vs[ri].length,
				// vprefs[ri].length, V+1);
				// System.out.println(vs[ri][vprefs[ri][V+1]]);
				b = b.putValue(rI + 1, vs[rI][vprefs[rI][V + 1]]);
			} else if (U >= maxValue) { // decrease the preference of an issue
										// to the next lowest level
				if (V <= 0) {
					continue;
				}
				b = b.putValue(rI + 1, vs[rI][vprefs[rI][V - 1]]);
			}
			U = utilitySpace.getUtility(b);
		}
		return b;
		// return preferredBids.get(rand.nextInt(preferredBids.size())).key;
	}

	// public Bid getMinBid() {
	// if(minBid == null) {
	// minBid = Collections.min(preferredBids).key;
	// }
	// return minBid;
	// }

	// public void genPreferredBids(int count) throws Exception {
	// Bid best = utilitySpace.getMaxUtilityBid();
	// genIssuePreferenceOrder();
	// System.out.println(Arrays.toString(iprefs));
	// System.out.println(Arrays.toString(vprefs));
	// maxCount = count;
	// preferredBids = genBids(best, 0);
	// }

	// int maxCount;
	// int count = 0;
	// private ArrayList<Tuple<Bid,Double>> genBids(Bid b, int i) throws
	// Exception {
	// ArrayList<Tuple<Bid,Double>> bids = new ArrayList<Tuple<Bid,Double>>();
	// int I = iprefs[i];
	// int M = vprefs[I].length;
	// if(i < iprefs.length-1) {
	// for(int j = M-1; j >= 0 && count < maxCount; j--) {
	// Bid b1 = new Bid(b);
	// Value v =
	// ((IssueDiscrete)utilitySpace.getDomain().getIssue(I)).getValue(vprefs[I][j]);
	// b1.setValue(I+1,v);
	// bids.addAll(genBids(b1, i + 1));
	// }
	// }
	// else {
	// for(int j = M-1; j >= 0 && count < maxCount; j--) {
	// Bid b1 = new Bid(b);
	// Value v =
	// ((IssueDiscrete)utilitySpace.getDomain().getIssue(I)).getValue(vprefs[I][j]);
	// b1.setValue(I+1,v);
	// bids.add(new Tuple<Bid,Double>(b1,utilitySpace.getUtility(b1)));
	// count++;
	// }
	// }
	// return bids;
	// }
}
