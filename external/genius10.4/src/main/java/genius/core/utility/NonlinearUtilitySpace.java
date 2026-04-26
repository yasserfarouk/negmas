package genius.core.utility;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.InvalidPropertiesFormatException;

import genius.core.Bid;
import genius.core.Domain;
import genius.core.DomainImpl;
import genius.core.xml.SimpleDOMParser;
import genius.core.xml.SimpleElement;

/**
 * 
 * In the non-linear scenarios, the agents no longer have linear utility
 * functions; instead, they can only sample their utility of a bid through the
 * getUtility() method. The utility of an offer can be completely arbitrary, so
 * there is no underlying structure you can use. In terms of the API of Genius,
 * this means the agents no longer have access to methods pertaining to linear
 * scenarios (e.g., getWeight()). Please use the getUtility() method to sample
 * the utilities in the non-linear scenarios to search through the outcome
 * space.
 */
@SuppressWarnings("serial")
public class NonlinearUtilitySpace extends AbstractUtilitySpace {
	private double maxUtilityValue;
	private UtilityFunction nonlinearFunction;
	private ArrayList<InclusiveHyperRectangle> allinclusiveConstraints;
	private ArrayList<ExclusiveHyperRectangle> allexclusiveConstraints;
	private SimpleElement fXMLRoot;

	/**
	 * Creates an empty nonlinear utility space.
	 */
	public NonlinearUtilitySpace() {
		super(new DomainImpl());
		this.nonlinearFunction = new UtilityFunction();
		this.allinclusiveConstraints = new ArrayList<InclusiveHyperRectangle>();
		this.allexclusiveConstraints = new ArrayList<ExclusiveHyperRectangle>();
	}

	public NonlinearUtilitySpace(Domain domain) {
		super(domain);
		this.nonlinearFunction = new UtilityFunction();
		this.allinclusiveConstraints = new ArrayList<InclusiveHyperRectangle>();
		this.allexclusiveConstraints = new ArrayList<ExclusiveHyperRectangle>();
	}

	public NonlinearUtilitySpace(Domain domain, String fileName)
			throws Exception {
		super(domain);
		this.nonlinearFunction = new UtilityFunction();
		this.fileName = fileName;
		this.allinclusiveConstraints = new ArrayList<InclusiveHyperRectangle>();
		this.allexclusiveConstraints = new ArrayList<ExclusiveHyperRectangle>();

		if (!fileName.equals("")) {
			SimpleDOMParser parser = new SimpleDOMParser();
			BufferedReader file = new BufferedReader(
					new FileReader(new File(fileName)));
			SimpleElement root = parser.parse(file);
			fXMLRoot = root;
			loadNonlinearSpace(root);

		} else
			throw new IOException();
	}

	private ArrayList<InclusiveHyperRectangle> getAllInclusiveConstraints() {
		return this.allinclusiveConstraints;
	}

	private ArrayList<ExclusiveHyperRectangle> getAllExclusiveConstraints() {
		return this.allexclusiveConstraints;
	}

	/** create a clone of another utility space */
	public NonlinearUtilitySpace(NonlinearUtilitySpace us) {
		super(us.getDomain());
		fileName = us.getFileName();
		fXMLRoot = us.fXMLRoot;
		maxUtilityValue = us.getMaxUtilityValue();
		nonlinearFunction = us.getNonlinearFunction();
		allinclusiveConstraints = us.getAllInclusiveConstraints();
		allexclusiveConstraints = us.getAllExclusiveConstraints();
		this.setDiscount(us.getDiscountFactor());
		this.setReservationValue(us.getReservationValue());
	}

	/**
	 * parse xml file and load nonlinear utility space
	 * 
	 * @param rectangeElements
	 * @return
	 */
	private ArrayList<Constraint> loadHyperRectangles(
			Object[] rectangeElements) {

		ArrayList<Constraint> hyperRectangleConstraints = new ArrayList<Constraint>();

		for (int j = 0; j < rectangeElements.length; j++) {

			HyperRectangle rectangle = null;
			ArrayList<Bound> boundlist = new ArrayList<Bound>();
			Object[] bounds = null;

			if (((SimpleElement) rectangeElements[j])
					.getChildByTagName("INCLUDES").length != 0) {
				rectangle = new InclusiveHyperRectangle();
				allinclusiveConstraints
						.add((InclusiveHyperRectangle) rectangle);
				bounds = ((SimpleElement) rectangeElements[j])
						.getChildByTagName("INCLUDES");
			}

			if (((SimpleElement) rectangeElements[j])
					.getChildByTagName("EXCLUDES").length != 0) {
				rectangle = new ExclusiveHyperRectangle();
				allexclusiveConstraints
						.add((ExclusiveHyperRectangle) rectangle);
				bounds = ((SimpleElement) rectangeElements[j])
						.getChildByTagName("EXCLUDES");
			}

			if ((((SimpleElement) rectangeElements[j])
					.getChildByTagName("INCLUDES").length == 0)
					&& (((SimpleElement) rectangeElements[j])
							.getChildByTagName("EXCLUDES").length == 0)) {
				rectangle = new InclusiveHyperRectangle(true);
			} else {
				for (int k = 0; k < bounds.length; k++) {
					Bound b = new Bound(
							((SimpleElement) bounds[k]).getAttribute("index"),
							((SimpleElement) bounds[k]).getAttribute("min"),
							((SimpleElement) bounds[k]).getAttribute("max"));
					boundlist.add(b);
				}
				rectangle.setBoundList(boundlist);
			}

			rectangle.setUtilityValue(
					Double.parseDouble(((SimpleElement) rectangeElements[j])
							.getAttribute("utility")));
			if (((SimpleElement) rectangeElements[j])
					.getAttribute("weight") != null)
				rectangle.setWeight(
						Double.parseDouble(((SimpleElement) rectangeElements[j])
								.getAttribute("weight")));

			hyperRectangleConstraints.add(rectangle);
		}
		return hyperRectangleConstraints;

	}

	private UtilityFunction loadUtilityFunction(SimpleElement utility) {

		UtilityFunction currentFunction = new UtilityFunction();
		currentFunction.setAggregationType(AGGREGATIONTYPE
				.getAggregationType(utility.getAttribute("aggregation")));
		if (utility.getAttribute("weight") != null)
			currentFunction.setWeight(
					Double.parseDouble(utility.getAttribute("weight")));

		// similarly other constraint can be parsed and add to the constraints
		// by adding a addConstraints method
		currentFunction.setConstraints(loadHyperRectangles(
				utility.getChildByTagName("hyperRectangle")));

		Object[] innerFunctions = utility.getChildByTagName("ufun");

		for (int k = 0; k < innerFunctions.length; k++) {
			currentFunction.addUtilityFunction(
					loadUtilityFunction(((SimpleElement) innerFunctions[k])));
		}
		return currentFunction;
	}

	@Override
	public SimpleElement toXML() throws IOException {
		/**
		 * Instead of conversion, we read in the existing XML file from disk,
		 * plug in the reservation value and discount factor and return that.
		 */
		SimpleElement root;
		SimpleDOMParser parser = new SimpleDOMParser();
		BufferedReader file = new BufferedReader(
				new FileReader(new File(fileName)));
		root = parser.parse(file);

		Object[] reservationelts = root.getChildByTagName("reservation");
		if (reservationelts == null || reservationelts.length == 0
				|| !(reservationelts[0] instanceof SimpleElement)) {
			throw new InvalidPropertiesFormatException(
					"file does not contain reservation value");
		}
		Object[] discountelts = root.getChildByTagName("discount_factor");
		if (discountelts == null || discountelts.length == 0
				|| !(discountelts[0] instanceof SimpleElement)) {
			throw new InvalidPropertiesFormatException(
					"file does not contain discount factor value");
		}

		((SimpleElement) reservationelts[0]).setAttribute("value",
				"" + getReservationValueUndiscounted());

		((SimpleElement) discountelts[0]).setAttribute("value",
				"" + getDiscountFactor());

		return root;

	}

	private void loadNonlinearSpace(SimpleElement root) {

		// load reservation value
		try {
			if ((root.getChildByTagName("reservation") != null)
					&& (root.getChildByTagName("reservation").length > 0)) {
				SimpleElement xml_reservation = (SimpleElement) (root
						.getChildByTagName("reservation")[0]);
				this.setReservationValue(
						Double.valueOf(xml_reservation.getAttribute("value")));
				System.out.println(
						"Reservation value: " + this.getReservationValue());
			}
		} catch (Exception e) {
			System.out.println("Utility space has no reservation value");
		}
		// load discount factor
		try {
			if ((root.getChildByTagName("discount_factor") != null)
					&& (root.getChildByTagName("discount_factor").length > 0)) {
				SimpleElement xml_reservation = (SimpleElement) (root
						.getChildByTagName("discount_factor")[0]);
				double df = Double
						.parseDouble(xml_reservation.getAttribute("value"));
				this.setDiscount(validateDiscount(df));
				System.out
						.println("Discount value: " + this.getDiscountFactor());
			}
		} catch (Exception e) {
			System.out.println("Utility space has no discount factor;");
		}

		// load utility
		Object utility = ((SimpleElement) root.getChildElements()[0])
				.getChildByTagName("utility")[0];
		this.setMaxUtilityValue(Double.parseDouble(
				((SimpleElement) utility).getAttribute("maxutility")));
		this.nonlinearFunction = loadUtilityFunction(
				(SimpleElement) ((SimpleElement) utility)
						.getChildByTagName("ufun")[0]);
	}

	private double getMaxUtilityValue() {
		return maxUtilityValue;
	}

	private void setMaxUtilityValue(double maxUtilityValue) {
		this.maxUtilityValue = maxUtilityValue;
	}

	@Override
	public double getUtility(Bid bid) {
		double result = nonlinearFunction.getUtility(bid)
				/ this.maxUtilityValue;

		if (result > 1)
			return 1;
		else
			return result;
	}

	/**
	 * Uses the original equals of {@link Object}.
	 */
	@Override
	public boolean equals(Object obj) {
		return this == obj;
	}

	private UtilityFunction getNonlinearFunction() {
		return nonlinearFunction;
	}

	@Override
	public String toString() {

		String result = "";
		for (InclusiveHyperRectangle rec : this.getAllInclusiveConstraints()) {
			ArrayList<Bound> boundList = rec.getBoundList();
			result += ("Rectangle: \n");
			for (Bound bound : boundList) {
				result += ("Issue with index " + bound.getIssueIndex() + " min:"
						+ bound.getMin() + " max:" + bound.getMax() + " \n");
			}

		}
		return result;
	}

	@Override
	public UtilitySpace copy() {
		return new NonlinearUtilitySpace(this);
	}

	@Override
	public String isComplete() {
		return null;
	}
}
