package genius.core.utility;

import genius.core.Bid;
import genius.core.Domain;
import genius.core.timeline.TimeLineInfo;
import genius.core.timeline.Timeline;

/**
 * Implements the basic functionality of {@link UtilitySpace} but does not
 * implement the details. Adds the discountFactor as a mechanism to implement
 * discount. A filename is remembered. Also adds default functionality to
 * support implementation of concrete utility spaces.
 */
@SuppressWarnings("serial")
public abstract class AbstractUtilitySpace implements UtilitySpace {

	public static final String DISCOUNT_FACTOR = "discount_factor";
	public static final String RESERVATION = "reservation";

	private Domain domain;
	protected String fileName;
	private double discountFactor = 1;
	private Double fReservationValue = null;
	private UtilitySpaceTools ustools;

	/**
	 * sets domain and tries to load the file into XML root.
	 * 
	 * @param dom
	 *            the {@link Domain} to load
	 */
	public AbstractUtilitySpace(Domain dom) {
		domain = dom;
		ustools = new UtilitySpaceTools(this);
	}

	@Override
	public Domain getDomain() {
		return domain;
	}

	/**
	 * @param newRV
	 *            new reservation value.
	 */
	public void setReservationValue(double newRV) {
		fReservationValue = newRV;
	}

	/**
	 * @param newDiscount
	 *            new discount factor.
	 */
	public void setDiscount(double newDiscount) {
		discountFactor = validateDiscount(newDiscount);
	}

	/**
	 * The reservation value is the least favourable point at which one will
	 * accept a negotiated agreement. Also sometimes referred to as the walk
	 * away point.
	 * <p>
	 * This is value remains constant during the negotiation. However, by
	 * default, the reservation value descreases with time. To obtain the
	 * discounted version of the reservation value, use
	 * {@link #getReservationValueWithDiscount(TimeLineInfo)}.
	 * 
	 * @return undiscounted reservation value of the preference profile (may be
	 *         null).
	 */
	public Double getReservationValue() {
		return getReservationValueUndiscounted();
	}

	/**
	 * Equivalent to {@link #getReservationValue()}, but always returns a double
	 * value. When the original reservation value is <b>null</b> it returns the
	 * default value 0.
	 * 
	 * @return undiscounted reservation value of the preference profile (never
	 *         null).
	 * @see #getReservationValue()
	 */
	public double getReservationValueUndiscounted() {
		if (fReservationValue == null)
			return 0;
		return fReservationValue;
	}

	/**
	 * The discounted version of {@link #getReservationValue()}.
	 * 
	 * @param time
	 *            at which we want to know the utility of the reservation value.
	 * @return discounted reservation value.
	 */
	public double getReservationValueWithDiscount(double time) {
		Double rv = getReservationValue();
		if (rv == null || rv == 0)
			return 0;

		return discount(rv, time);
	}

	/**
	 * The discounted version of {@link #getReservationValue()}.
	 * 
	 * @param timeline
	 *            specifying the current time in the negotiation.
	 * @return discounted reservation value.
	 */
	public double getReservationValueWithDiscount(TimeLineInfo timeline) {
		return getReservationValueWithDiscount(timeline.getTime());
	}

	/**
	 * @return true if the domain features discounts.
	 */
	public boolean isDiscounted() {
		return discountFactor < 1.0;
	}

	/**
	 * @return Discount factor of this preference profile.
	 */
	public final double getDiscountFactor() {
		return discountFactor;
	}

	/**
	 * @return filename of this preference profile.
	 */
	public String getFileName() {
		return fileName;
	}

	/**
	 * Let d in (0, 1) be the discount factor. (If d &le; 0 or d &ge; 1, we
	 * assume that d = 1.) Let t in [0, 1] be the current time, as defined by
	 * the {@link Timeline}. We compute the <i>discounted</i> utility
	 * discountedUtility as follows:
	 * 
	 * discountedUtility = originalUtility * d^t.
	 * 
	 * For t = 0 the utility remains unchanged, and for t = 1 the original
	 * utility is multiplied by the discount factor. The effect is almost linear
	 * in between. Works with any utility space.
	 * 
	 * @param bid
	 *            of which we are interested in its utility.
	 * @param timeline
	 *            indicating the time passed in the negotiation.
	 * @return discounted utility.
	 */
	public double getUtilityWithDiscount(Bid bid, TimeLineInfo timeline) {
		double time = timeline.getTime();
		return getUtilityWithDiscount(bid, time);
	}

	/**
	 * See {@link #getUtilityWithDiscount(Bid, double)}.
	 * 
	 * @param bid
	 *            of which we want to know the utility at the given time.
	 * @param time
	 *            at which we want to know the utility of the bid.
	 * @return discounted utility.
	 */
	public double getUtilityWithDiscount(Bid bid, double time) {
		double util = 0;
		try {
			util = getUtility(bid);
		} catch (Exception e) {
			e.printStackTrace();
		}

		double discountedUtil = discount(util, time);
		return discountedUtil;
	}

	/**
	 * Specific implementation for discount, based on a discount factor.
	 * Computes:
	 * 
	 * discountedUtil = util * Math.pow(discount, time).
	 * 
	 * Checks for bounds on the discount factor and time.
	 */
	@Override
	public Double discount(double util, double time) {
		return discount(util, time, discountFactor);
	}

	/**
	 * Computes:
	 * 
	 * discountedUtil = util * Math.pow(discount, time).
	 * 
	 * Checks for bounds on the discount factor and time.
	 * 
	 * @param util
	 *            undiscounted utility.
	 * @param time
	 *            at which we want to know the discounted utility.
	 * @param discountFactor
	 *            of the preference profile.
	 * @return discounted version of the given utility at the given time.
	 */
	private double discount(double util, double time, double discountFactor) {
		double discount = discountFactor;
		if (time < 0) {
			System.err.println("Warning: time = " + time
					+ " < 0, using time = 0 instead.");
			time = 0;
		}
		if (time > 1) {
			System.err.println("Warning: time = " + time
					+ " > 1, using time = 1 instead.");
			time = 1;
		}

		double discountedUtil = util * Math.pow(discount, time);
		return discountedUtil;
	}

	protected double validateDiscount(double df) {
		if (df < 0 || df > 1) {
			System.err.println(
					"Warning: discount factor = " + df + " was discarded.");
		}

		if (df <= 0 || df > 1) {
			df = 1;
		}
		return df;
	}

	/**
	 * Returns the maximum bid in the utility space. This is only supported for
	 * linear utility spaces. Totally revised, brute-force search now.
	 * 
	 * @return a bid with the maximum utility value attainable in this util
	 *         space
	 * @throws Exception
	 *             if there is no bid at all in this util space.
	 */
	public final Bid getMaxUtilityBid() throws Exception {
		return ustools.getMaxUtilityBid();
	}

	/**
	 * Returns the worst bid in the utility space. This is only supported for
	 * linear utility spaces.
	 * 
	 * @return a bid with the lowest possible utility
	 * @throws Exception
	 *             if there is no bid at all in the util space
	 */
	public Bid getMinUtilityBid() throws Exception {
		return ustools.getMinUtilityBid();
	}
		
	/**
	 * Check if this utility space is ready for negotiation. To be so, the
	 * domain must match the given domain and the space must be complete.
	 * 
	 * @param dom
	 *            is the domain in which nego is taking place
	 * @throws Exception
	 *             if utility space is incomplete (@see isComplete())
	 */
	public void checkReadyForNegotiation(Domain dom) throws Exception {
		ustools.checkReadyForNegotiation(dom);
	}

	@Override
	public String getName() {
		if (fileName == null) {
			return "domain@" + Integer.toHexString(hashCode());
		}
		return fileName;
	}
}
