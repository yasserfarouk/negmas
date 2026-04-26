package genius.core.analysis.pareto;

import java.util.Collections;
import java.util.Set;

/**
 * Similar to BidPoint but can hold partial bids. Generic as it uses IssueValue
 *
 */
public class PartialBidPoint {

	/**
	 * This needs to be slightly negative, so that isDominatedBy also says yes
	 * if the other point is slightly negative of our point.
	 */
	private final static double MIN_DELTA = -0.0000000000001;
	private final Set<IssueValue> values;
	/**
	 * Caches the actual sum of the value utilities.
	 */
	private final Double utilA, utilB;

	public PartialBidPoint(Set<IssueValue> values, Double utilA, Double utilB) {
		this.values = values;
		this.utilA = utilA;
		this.utilB = utilB;
	}

	public Double utilA() {
		return utilA;
	}

	public Double utilB() {
		return utilB;
	}

	public Set<IssueValue> getValues() {
		return Collections.unmodifiableSet(values);
	}

	public boolean isDominatedBy(PartialBidPoint p) {
		return (p.utilA - utilA > MIN_DELTA) && (p.utilB - utilB > MIN_DELTA);
	}

	@Override
	public String toString() {
		return "point(" + utilA + "," + utilB + ")";
	}

}