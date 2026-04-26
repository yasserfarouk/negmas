package genius.core.analysis.pareto;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.IssueInteger;
import genius.core.issue.Value;
import genius.core.issue.ValueDiscrete;
import genius.core.issue.ValueInteger;
import genius.core.utility.AdditiveUtilitySpace;
import genius.core.utility.EvaluatorDiscrete;
import genius.core.utility.EvaluatorInteger;
import genius.core.utility.UtilitySpace;

/**
 * Algorithm to compute pareto fast.
 * <p>
 * 
 * <em>WARNING</em> this is unstable, experimental code. Use at your own risk.
 */
public class ParetoFrontierF {
	private final Set<PartialBidPoint> frontier = new HashSet<PartialBidPoint>();

	public ParetoFrontierF(List<Set<IssueValue>> valuesets) {
		init(valuesets);
	}

	/**
	 * Does the heavy plumbing job of getting valuesets from domain.
	 * 
	 * You can then get the frontier and convert back the Pareto points to
	 * BidPoints using <code>
	  		ArrayList<BidPoint> bidpoints = new ArrayList<BidPoint>();
		for (ParetoPoint point : fastparetof.getFrontier()) {
			bidpoints.add(new BidPoint(null, point.utilA(), point.utilB()));
		}</code>
	 * 
	 * 
	 * @throws Exception
	 * 
	 */
	public ParetoFrontierF(AdditiveUtilitySpace utilitySpaceA, AdditiveUtilitySpace utilitySpaceB) {

		List<Set<IssueValue>> valuesSets = new ArrayList<Set<IssueValue>>();
		for (Issue issue : utilitySpaceA.getDomain().getIssues()) {
			Set<IssueValue> valueSet = getValueSet(utilitySpaceA, utilitySpaceB, issue);
			valuesSets.add(valueSet);
		}
		init(valuesSets);
	}

	/**
	 * @return the set of {@link PartialBidPoint}s that comprise the pareto
	 *         frontier
	 */
	public Collection<PartialBidPoint> getFrontier() {
		return Collections.unmodifiableSet(frontier);
	}

	private void init(List<Set<IssueValue>> valuesets) {
		if (valuesets.isEmpty())
			throw new IllegalArgumentException("valuesets is empty");
		if (valuesets.size() == 1) {
			for (IssueValue issuevalue : valuesets.get(0)) {
				HashSet<IssueValue> values = new HashSet<>();
				values.add(issuevalue);
				add(new PartialBidPoint(values, issuevalue.getUtilityA(), issuevalue.getUtilityB()));
			}
		} else {
			int n = valuesets.size();
			List<Set<IssueValue>> subset1 = valuesets.subList(0, n / 2);
			List<Set<IssueValue>> subset2 = valuesets.subList(n / 2, n);
			merge(new ParetoFrontierF(subset1), new ParetoFrontierF(subset2));
		}
	}

	/**
	 * Constructor that merges two given frontiers.
	 * 
	 * @param f1
	 *            first pareto
	 * @param f2
	 *            second pareto
	 */
	private void merge(ParetoFrontierF f1, ParetoFrontierF f2) {
		for (PartialBidPoint point1 : f1.frontier) {
			for (PartialBidPoint point2 : f2.frontier) {
				Set<IssueValue> values = new HashSet<>(point1.getValues());
				values.addAll(point2.getValues());
				add(new PartialBidPoint(values, point1.utilA() + point2.utilA(), point1.utilB() + point2.utilB()));
			}
		}
	}

	/**
	 * Add newPoint if it is not yet dominated by the existing frontier.
	 * 
	 * @param newPoint
	 *            the {@link PartialBidPoint} to add
	 */
	private void add(PartialBidPoint newPoint) {
		// FIXME if we sort frontier we can avoid the search.
		for (PartialBidPoint p : frontier) {
			if (newPoint.isDominatedBy(p))
				return;
		}
		for (PartialBidPoint p : new HashSet<>(frontier)) {
			if (p.isDominatedBy(newPoint))
				frontier.remove(p);
		}
		frontier.add(newPoint);
	}

	/**
	 * Support func. Should maybe not be here??? Get the set of values for a
	 * given issue in the domain
	 * 
	 * @param utilitySpaceA
	 *            the first party {@link UtilitySpace}
	 * @param utilitySpaceB
	 *            the second party {@link UtilitySpace}
	 * @param issue
	 *            the issue to get the values for
	 * @return a {@link Set} of {@link IssueValue}s, one for each possible issue
	 *         value.
	 * @throws Exception
	 */
	private Set<IssueValue> getValueSet(AdditiveUtilitySpace utilitySpaceA, AdditiveUtilitySpace utilitySpaceB,
			Issue issue) {
		Set<IssueValue> valueSet = new HashSet<>();
		int nr = issue.getNumber();
		Double weightA = utilitySpaceA.getWeight(nr);
		Double weightB = utilitySpaceB.getWeight(nr);
		Double utilA, utilB;

		switch (issue.getType()) {
		case DISCRETE:
			IssueDiscrete issueDisc = (IssueDiscrete) issue;
			for (ValueDiscrete value : issueDisc.getValues()) {
				try {
					utilA = weightA * ((EvaluatorDiscrete) utilitySpaceA.getEvaluator(nr)).getEvaluation(value);
					utilB = weightB * ((EvaluatorDiscrete) utilitySpaceB.getEvaluator(nr)).getEvaluation(value);
				} catch (Exception e) {
					throw new IllegalArgumentException("failed to evaluate discrete issue " + issueDisc, e);
				}
				valueSet.add(new MyIssueValue(issue, value, utilA, utilB));
			}
			break;
		case INTEGER:
			IssueInteger issueInt = (IssueInteger) issue;

			for (int n = issueInt.getLowerBound(); n <= issueInt.getUpperBound(); n++) {
				ValueInteger value = new ValueInteger(n);
				utilA = weightA * ((EvaluatorInteger) utilitySpaceA.getEvaluator(nr)).getEvaluation(n);
				utilB = weightB * ((EvaluatorInteger) utilitySpaceB.getEvaluator(nr)).getEvaluation(n);
				valueSet.add(new MyIssueValue(issue, value, utilA, utilB));
			}
			break;
		default:
			throw new IllegalArgumentException("Unsupported issue type " + issue.getType());
		}
		return valueSet;
	}

}

class MyIssueValue implements IssueValue {
	private Issue issue;
	private Value value;
	private Double utilA;
	private Double utilB;

	public MyIssueValue(Issue issue, Value value, Double utilA, Double utilB) {
		this.issue = issue;
		this.value = value;
		this.utilA = utilA;
		this.utilB = utilB;
	}

	@Override
	public Double getUtilityA() {
		return utilA;
	}

	@Override
	public Double getUtilityB() {
		return utilB;
	}

	@Override
	public String toString() {
		return "<" + issue + " " + value + " " + utilA + " " + utilB + ">";
	}

	@Override
	public Issue getIssue() {
		return issue;
	}

	@Override
	public Value getValue() {
		return value;
	}

}
