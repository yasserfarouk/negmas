package genius.core.utility;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.InvalidPropertiesFormatException;

import genius.core.Bid;
import genius.core.Domain;
import genius.core.DomainImpl;
import genius.core.xml.SimpleDOMParser;
import genius.core.xml.SimpleElement;

@SuppressWarnings("serial")
public class ConstraintUtilitySpace extends AbstractUtilitySpace {

	private HashMap<Integer, Rank> rankingofIssues;
	private ArrayList<Integer> issueIndices;
	private HashMap<Integer, ArrayList<RConstraint>> contraintList;
	private ArrayList<ZeroOutcomeContraint> zeroOutcomeConstraints;

	public ConstraintUtilitySpace() {
		super(new DomainImpl());
		this.setRankingofIssues(new HashMap<Integer, Rank>());
		this.setContraints(new HashMap<Integer, ArrayList<RConstraint>>());
		this.setZeroOutcomeConstraints(new ArrayList<ZeroOutcomeContraint>());
	}

	public ConstraintUtilitySpace(Domain domain) {
		super(domain);
		this.setRankingofIssues(new HashMap<Integer, Rank>());
		this.setContraints(new HashMap<Integer, ArrayList<RConstraint>>());
		this.setIssueIndices(new ArrayList<Integer>());
		this.setZeroOutcomeConstraints(new ArrayList<ZeroOutcomeContraint>());
	}

	public ConstraintUtilitySpace(Domain domain, String fileName)
			throws IOException {
		super(domain);
		this.fileName = fileName;
		this.setRankingofIssues(new HashMap<Integer, Rank>());
		this.setContraints(new HashMap<Integer, ArrayList<RConstraint>>());
		this.setIssueIndices(new ArrayList<Integer>());
		this.setZeroOutcomeConstraints(new ArrayList<ZeroOutcomeContraint>());

		if (!fileName.equals("")) {
			SimpleDOMParser parser = new SimpleDOMParser();
			BufferedReader file = new BufferedReader(
					new FileReader(new File(fileName)));
			SimpleElement root = parser.parse(file);
			loadConstraintSpace(root);

		} else
			throw new IOException();
	}

	/** construct a clone of another utility space */
	public ConstraintUtilitySpace(ConstraintUtilitySpace us) {
		super(us.getDomain());
		fileName = us.getFileName();
		setRankingofIssues(us.getRankingofIssues());
		setIssueIndices(us.getIssueIndices());
		setContraints(us.getContraints());
		setZeroOutcomeConstraints(us.getZeroOutcomeConstraints());
		this.setDiscount(us.getDiscountFactor());
		this.setReservationValue(
				((ConstraintUtilitySpace) us).getReservationValue());
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

		Object[] reservationelts = root.getChildByTagName(RESERVATION);
		if (reservationelts == null || reservationelts.length == 0
				|| !(reservationelts[0] instanceof SimpleElement)) {
			throw new InvalidPropertiesFormatException(
					"file does not contain reservation value");
		}
		Object[] discountelts = root.getChildByTagName(DISCOUNT_FACTOR);
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

	private void loadRanks(SimpleElement ranks) {

		Object[] allRankObjects = ranks.getChildByTagName("issue");

		for (int k = 0; k < allRankObjects.length; k++) {

			this.issueIndices
					.add(Integer.valueOf(((SimpleElement) allRankObjects[k])
							.getAttribute("index").toString()));
			Rank currentRanking = new Rank(
					Integer.valueOf(((SimpleElement) allRankObjects[k])
							.getAttribute("index").toString()));

			Object[] rankList = ((SimpleElement) allRankObjects[k])
					.getChildByTagName("item");

			for (int m = 0; m < rankList.length; m++) {
				currentRanking.addRank(
						((SimpleElement) rankList[m]).getAttribute("value"),
						((SimpleElement) rankList[m]).getAttribute("rank"));
			}

			this.rankingofIssues
					.put(Integer.valueOf(((SimpleElement) allRankObjects[k])
							.getAttribute("index").toString()), currentRanking);
		}

	}

	private void addRConstraint(RConstraint currentConstraint) {

		ArrayList<RConstraint> myRConstraints;
		if (this.contraintList.containsKey(currentConstraint.getIssueIndex()))
			myRConstraints = this.contraintList
					.get(currentConstraint.getIssueIndex());
		else
			myRConstraints = new ArrayList<RConstraint>();

		myRConstraints.add(currentConstraint);
		this.contraintList.put(currentConstraint.getIssueIndex(),
				myRConstraints);
	}

	private void loadConstraints(SimpleElement constraints) {

		Object[] allConstraintObjects = constraints
				.getChildByTagName("constraint");

		for (int k = 0; k < allConstraintObjects.length; k++) {
			// check the type of the constraints and act accordingly..

			String constraintType = ((SimpleElement) allConstraintObjects[k])
					.getAttribute("type").toString();
			if ((constraintType.equals("inclusiveZeroOutcomeConstraint"))
					|| (constraintType
							.equals("conditionalZeroOutcomeConstraint"))) {

				ZeroOutcomeContraint currentConstraint;

				if (constraintType.equals("inclusiveZeroOutcomeConstraint"))
					currentConstraint = new InclusiveZeroOutcomeConstraint();
				else
					currentConstraint = new ConditionalZeroOutcomeConstraint();

				Object[] checkAssignmentList = ((SimpleElement) allConstraintObjects[k])
						.getChildByTagName("checkassignment");

				for (int s = 0; s < checkAssignmentList.length; s++) {
					currentConstraint.addContraint(
							Integer.valueOf(
									((SimpleElement) checkAssignmentList[s])
											.getAttribute("index").toString()),
							((SimpleElement) checkAssignmentList[s])
									.getAttribute("condition"));
				}
				this.zeroOutcomeConstraints.add(currentConstraint);
			} // if
			else if ((constraintType.equals("zeroConstraint"))
					|| (constraintType.equals("sumZeroConstraint"))
					|| (constraintType.equals("sumZeroNotConstraint"))
					|| (constraintType.equals("sumZeroConstraintList"))
					|| (constraintType.equals("conditionalZeroConstraint"))) {

				RConstraint currentConstraint = null;

				if (constraintType.equals("zeroConstraint")) {
					currentConstraint = new ZeroConstraint(Integer
							.valueOf(((SimpleElement) allConstraintObjects[k])
									.getAttribute("index").toString()));
					Object[] valueAssignmentList = ((SimpleElement) allConstraintObjects[k])
							.getChildByTagName("valueassignment");
					for (int s = 0; s < valueAssignmentList.length; s++) {
						currentConstraint.addContraint(Integer.valueOf(
								((SimpleElement) valueAssignmentList[s])
										.getAttribute("index").toString()),
								((SimpleElement) valueAssignmentList[s])
										.getAttribute("value"));
					}

				} else if (constraintType.equals("conditionalZeroConstraint")) {

					currentConstraint = new ConditionalZeroConstraint(
							Integer.valueOf(
									((SimpleElement) allConstraintObjects[k])
											.getAttribute("index").toString()),
							((SimpleElement) allConstraintObjects[k])
									.getAttribute("value").toString());
					Object[] valueAssignmentList = ((SimpleElement) allConstraintObjects[k])
							.getChildByTagName("valueassignment");
					for (int s = 0; s < valueAssignmentList.length; s++) {
						currentConstraint.addContraint(Integer.valueOf(
								((SimpleElement) valueAssignmentList[s])
										.getAttribute("index").toString()),
								((SimpleElement) valueAssignmentList[s])
										.getAttribute("value"));
					}

				} else if ((constraintType.equals("sumZeroConstraint"))
						|| (constraintType.equals("sumZeroNotConstraint"))) {

					if (constraintType.equals("sumZeroConstraint"))
						currentConstraint = new SumZeroConstraint(
								Integer.valueOf(
										((SimpleElement) allConstraintObjects[k])
												.getAttribute("index")
												.toString()));
					else
						currentConstraint = new SumZeroNotConstraint(
								Integer.valueOf(
										((SimpleElement) allConstraintObjects[k])
												.getAttribute("index")
												.toString()));

					((SumZeroConstraint) currentConstraint).setValueToBeChecked(
							((SimpleElement) allConstraintObjects[k])
									.getAttribute("value").toString());
					((SumZeroConstraint) currentConstraint).setMax(Integer
							.valueOf(((SimpleElement) allConstraintObjects[k])
									.getAttribute("max").toString()));
					((SumZeroConstraint) currentConstraint).setMin(Integer
							.valueOf(((SimpleElement) allConstraintObjects[k])
									.getAttribute("min").toString()));

					Object[] issueList = ((SimpleElement) allConstraintObjects[k])
							.getChildByTagName("item");

					for (int t = 0; t < issueList.length; t++)
						((SumZeroConstraint) currentConstraint).addRelatedIssue(
								Integer.valueOf(((SimpleElement) issueList[t])
										.getAttribute("index").toString()));

				} else if (constraintType.equals("sumZeroConstraintList")) {

					currentConstraint = new SumZeroConstraintList(Integer
							.valueOf(((SimpleElement) allConstraintObjects[k])
									.getAttribute("index").toString()));

					((SumZeroConstraintList) currentConstraint).setMax(Integer
							.valueOf(((SimpleElement) allConstraintObjects[k])
									.getAttribute("max").toString()));
					((SumZeroConstraintList) currentConstraint).setMin(Integer
							.valueOf(((SimpleElement) allConstraintObjects[k])
									.getAttribute("min").toString()));

					Object[] issueList = ((SimpleElement) allConstraintObjects[k])
							.getChildByTagName("item");

					for (int t = 0; t < issueList.length; t++)
						((SumZeroConstraintList) currentConstraint)
								.addRelatedIssue(Integer
										.valueOf(((SimpleElement) issueList[t])
												.getAttribute("index")
												.toString()));

					Object[] conditionList = ((SimpleElement) allConstraintObjects[k])
							.getChildByTagName("condition");

					for (int t = 0; t < conditionList.length; t++)
						((SumZeroConstraintList) currentConstraint)
								.addValueToBeChecked(
										((SimpleElement) conditionList[t])
												.getAttribute("value")
												.toString());

				}
				addRConstraint(currentConstraint);
			}

		} // for

	}

	private void loadConstraintSpace(SimpleElement root) {
		// load reservation value
		try {
			if ((root.getChildByTagName(RESERVATION) != null)
					&& (root.getChildByTagName(RESERVATION).length > 0)) {
				SimpleElement xml_reservation = (SimpleElement) (root
						.getChildByTagName(RESERVATION)[0]);
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
			if ((root.getChildByTagName(DISCOUNT_FACTOR) != null)
					&& (root.getChildByTagName(DISCOUNT_FACTOR).length > 0)) {
				SimpleElement xmlReservation = (SimpleElement) (root
						.getChildByTagName(DISCOUNT_FACTOR)[0]);
				double df = Double
						.parseDouble(xmlReservation.getAttribute("value"));
				this.setDiscount(validateDiscount(df));
				System.out
						.println("Discount value: " + this.getDiscountFactor());
			}
		} catch (Exception e) {
			System.out.println("Utility space has no discount factor;");
		}

		// load utility
		Object rules = ((SimpleElement) root.getChildElements()[0])
				.getChildByTagName("rules")[0];
		// load the ranks
		loadRanks((SimpleElement) ((SimpleElement) rules)
				.getChildByTagName("ranks")[0]);
		loadConstraints((SimpleElement) ((SimpleElement) rules)
				.getChildByTagName("constraints")[0]);
		// similarly load the constraints
	}

	private HashMap<Integer, Rank> getRankingofIssues() {
		return rankingofIssues;
	}

	private void setRankingofIssues(HashMap<Integer, Rank> rankingofIssues) {
		this.rankingofIssues = rankingofIssues;
	}

	private HashMap<Integer, ArrayList<RConstraint>> getContraints() {
		return contraintList;
	}

	private void setContraints(
			HashMap<Integer, ArrayList<RConstraint>> contraints) {
		this.contraintList = contraints;
	}

	private boolean isZeroUtility(Integer index, Bid bid) {

		if (!this.contraintList.containsKey(index))
			return false;

		for (RConstraint myConstraints : this.contraintList.get(index)) {
			if (myConstraints.willZeroUtility(bid))
				return true;
		}
		return false;
	}

	@Override
	public double getUtility(Bid bid) {

		double utility = 0.0;

		for (int s = 0; s < this.zeroOutcomeConstraints.size(); s++) {
			if (zeroOutcomeConstraints.get(s).willGetZeroOutcomeUtility(bid))
				return 0.0;
		}

		for (int i = 0; i < this.issueIndices.size(); i++) {

			if (!isZeroUtility(this.issueIndices.get(i), bid))
				utility += this.rankingofIssues.get(this.issueIndices.get(i))
						.getNormalizedRank(bid
								.getValue(this.issueIndices.get(i)).toString());
		}
		return ((double) ((double) utility / this.issueIndices.size()));

	}

	/**
	 * Uses the original equals of {@link Object}.
	 */
	@Override
	public boolean equals(Object obj) {
		return this == obj;
	}

	private ArrayList<Integer> getIssueIndices() {
		return issueIndices;
	}

	private void setIssueIndices(ArrayList<Integer> issueIndices) {
		this.issueIndices = issueIndices;
	}

	public ArrayList<ZeroOutcomeContraint> getZeroOutcomeConstraints() {
		return zeroOutcomeConstraints;
	}

	public void setZeroOutcomeConstraints(
			ArrayList<ZeroOutcomeContraint> zeroOutcomeConstraints) {
		this.zeroOutcomeConstraints = zeroOutcomeConstraints;
	}

	@Override
	public UtilitySpace copy() {
		return new ConstraintUtilitySpace(this);
	}

	@Override
	public String isComplete() {
		return null;
	}

}
