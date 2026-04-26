package genius.core.utility;

import java.util.List;

import java.util.Enumeration;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.Vector;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

import genius.core.Bid;
import genius.core.Domain;
import genius.core.DomainImpl;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.Objective;
import genius.core.xml.SimpleDOMParser;
import genius.core.xml.SimpleElement;

/**
 * The additive utility space couples all objectives to weights and evaluators.
 * 
 * @author D. Tykhonov, K. Hindriks, W. Pasman
 */

public class AdditiveUtilitySpace extends AbstractUtilitySpace {

	/**
	 * 
	 */
	private static final long serialVersionUID = 8746748105840831474L;

	private final Map<Objective, Evaluator> fEvaluators;

	/**
	 * Creates an empty utility space.
	 */
	public AdditiveUtilitySpace() {
		this(new DomainImpl(), new HashMap<Objective, Evaluator>());
	}

	/**
	 * Creates a new utilityspace of the given domain.
	 * 
	 * @param domain
	 *            for which the utilityspace should be specified.
	 */
	public AdditiveUtilitySpace(Domain domain) {
		this(domain, new HashMap<Objective, Evaluator>());
	}

	public AdditiveUtilitySpace(Domain domain,
			Map<Objective, Evaluator> fEvaluators) {
		super(domain);
		this.fEvaluators = fEvaluators;
		normalizeWeights();
	}

	/**
	 * Create new default util space for a given domain.
	 * 
	 * @param domain
	 * @param fileName
	 *            to read domain from. Set fileName to "" if no file available,
	 *            in which case default evaluators are loaded..
	 * @throws IOException
	 *             if error occurs, e.g. if domain does not match the util
	 *             space, or file not found.
	 */
	public AdditiveUtilitySpace(Domain domain, String fileName)
			throws IOException {
		super(domain);
		this.fileName = fileName;
		fEvaluators = new HashMap<Objective, Evaluator>();
		if (!fileName.equals(""))
			loadTreeFromFile(fileName);
		else { // add evaluator to all objectives
			List<Objective> objectives = domain.getObjectives();
			for (Objective obj : objectives) {
				Evaluator eval = defaultEvaluator(obj);
				fEvaluators.put(obj, eval);
			}

		}
	}

	/**
	 * Copies the data from another UtilitySpace.
	 * 
	 * @param us
	 *            utility space to be cloned.
	 */
	public AdditiveUtilitySpace(AdditiveUtilitySpace us) {
		super(us.getDomain());
		fileName = us.getFileName();
		fEvaluators = new HashMap<Objective, Evaluator>();
		setReservationValue(us.getReservationValue());
		// and clone the evaluators
		for (Objective obj : getDomain().getObjectives()) {
			Evaluator e = us.getEvaluator(obj.getNumber());
			if (e != null)
				fEvaluators.put(obj, e.clone());
			/*
			 * else incomplete. But that seems allowed. especially, objectives
			 * (the non-Issues) won't generally have an evlauator
			 */
		}
		setDiscount(us.getDiscountFactor());
	}

	/**************** implements UtilitySpace ****************/

	@Override
	public double getUtility(Bid bid) {
		double utility = 0;

		Objective root = getDomain().getObjectivesRoot();
		Enumeration<Objective> issueEnum = root.getPreorderIssueEnumeration();
		while (issueEnum.hasMoreElements()) {
			Objective is = issueEnum.nextElement();
			Evaluator eval = fEvaluators.get(is);
			if (eval == null) {
				throw new IllegalArgumentException(
						"UtilitySpace does not contain evaluator for issue "
								+ is + ". ");
			}

			switch (eval.getType()) {
			case DISCRETE:
			case INTEGER:
			case REAL:
				utility += eval.getWeight()
						* eval.getEvaluation(this, bid, is.getNumber());
				break;
			case OBJECTIVE:
				// we ignore OBJECTIVE. Not clear what it is and why.
				break;
			}
		}
		double result = utility;
		if (result > 1)
			return 1;
		return result;
	}

	@Override
	public SimpleElement toXML() throws IOException {
		SimpleElement root = (getDomain().getObjectivesRoot()).toXML();
		root = toXMLrecurse(root);
		SimpleElement rootWrapper = new SimpleElement("utility_space");
		rootWrapper.addChildElement(root);

		SimpleElement discountFactor = new SimpleElement("discount_factor");
		discountFactor.setAttribute("value", getDiscountFactor() + "");
		rootWrapper.addChildElement(discountFactor);

		SimpleElement reservationValue = new SimpleElement(RESERVATION);
		reservationValue.setAttribute("value",
				getReservationValueUndiscounted() + "");
		rootWrapper.addChildElement(reservationValue);

		return rootWrapper;
	}

	@Override
	public String isComplete() {
		/**
		 * This function *should* check that the domainSubtreeP is a subtree of
		 * the utilSubtreeP, and that all leaf nodes are complete. However
		 * currently we only check that all the leaf nodes are complete, We
		 * don't have the domain template here anymore. so we can only check
		 * that all fields are filled.
		 */
		List<Issue> issues = getDomain().getIssues();
		if (issues == null)
			return "Utility space is not complete, in fact it is empty!";
		String mess;
		for (Issue issue : issues) {
			Evaluator ev = getEvaluator(issue.getNumber());
			if (ev == null)
				return "issue " + issue.getName() + " has no evaluator";
			mess = (ev.isComplete(issue));
			if (mess != null)
				return mess;
		}
		return null;
	}

	@Override
	public String toString() {
		String result = "";
		for (Entry<Objective, Evaluator> entry : fEvaluators.entrySet()) {
			result += ("Issue weight " + entry.getValue().getWeight() + "\n");
			result += ("Values " + entry.getKey().getName() + ": "
					+ entry.getValue().toString() + "\n");
		}
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (!(obj instanceof AdditiveUtilitySpace))
			return false;
		AdditiveUtilitySpace obj2 = (AdditiveUtilitySpace) obj;
		if (!getDomain().equals(obj2.getDomain()))
			return false;
		for (Entry<Objective, Evaluator> entry : fEvaluators.entrySet()) {
			Evaluator eval2 = obj2.getEvaluator(entry.getKey().getNumber());
			if (!entry.getValue().equals(eval2)) {
				return false;
			}
		}
		return true;
	}

	@Override
	public UtilitySpace copy() {
		return new AdditiveUtilitySpace(this);
	}

	/************ Additive-specific functions *****************/

	/**
	 * @return number of issues. This can only be used for linear utility
	 *         functions.
	 */
	public final int getNrOfEvaluators() {
		return fEvaluators.size();
	}

	/**
	 * Returns the evaluator of an issue for the given index. This can only be
	 * used for linear utility functions.
	 * 
	 * @param index
	 *            The IDnumber of the Objective or Issue
	 * @return An Evaluator for the Objective or Issue.
	 */
	public Evaluator getEvaluator(int index) {
		Objective obj = getDomain().getObjectivesRoot().getObjective(index);
		if (obj != null) {
			return fEvaluators.get(obj);
		} else
			return null;
	}

	/**
	 * @param obj
	 *            the objective for which an evaluator is needed
	 * @return evaluator for given objective, or null if no such objective.
	 */
	public Evaluator getEvaluator(Objective obj) {
		return fEvaluators.get(obj);
	}

	/**
	 * Returns the utility of one issue in the bid. Note that this value is in
	 * the range [0,1] as it is not normalized by the issue weight. Only works
	 * with linear utility spaces.
	 * 
	 * @param pIssueIndex
	 *            of the issue.
	 * @param bid
	 * @return evaluation of the value of the issue of the given bid.
	 */
	public final double getEvaluation(int pIssueIndex, Bid bid) {
		Object lObj = getDomain().getObjectivesRoot().getObjective(pIssueIndex);
		Evaluator lEvaluator = fEvaluators.get(lObj);

		return lEvaluator.getEvaluation(this, bid, pIssueIndex);
	}

	/**
	 * 
	 * @param issueID
	 *            The Issue or Objective to get the weight from
	 * @return The weight, or 0 if the objective doesn't exist. Only works with
	 *         linear utility spaces.
	 */
	public double getWeight(int issueID) 
	{
		Objective ob = getDomain().getObjectivesRoot().getObjective(issueID);
		if (ob != null) {
			Evaluator ev = fEvaluators.get(ob);
			if (ev != null) {
				return ev.getWeight();
			}
		}

//		System.out.println("Objective of issue #" + issueID + " not found. Weight 0 returned.");
		return 0.0;
	}

	public double getWeight(Objective obj) {

		Evaluator ev = fEvaluators.get(obj);
		if (ev != null) {
			return ev.getWeight();
		} else
			return 0.0;
	}

	/**
	 * Method used to set the weight of the given objective. Only works if the
	 * objective has been unlockeed.
	 * 
	 * @param objective
	 *            of which the weights must be set.
	 * @param weight
	 *            to which the weight of the objective must be set.
	 * @return the new weight of the issue after normalization.
	 */
	public double setWeight(Objective objective, double weight) {
		try {
			Evaluator ev = fEvaluators.get(objective);
			double oldWt = ev.getWeight();
			if (!ev.weightLocked()) {
				ev.setWeight(weight); // set weight
			}
			this.normalizeChildren(objective.getParent());
			if (this.checkTreeNormalization()) {
				return fEvaluators.get(objective).getWeight();
			} else {
				ev.setWeight(oldWt); // set the old weight back.
				return fEvaluators.get(objective).getWeight();
			}
		} catch (NullPointerException npe) {
			return -1;
		}
	}

	public void setWeights(List<Issue> issues, double[] weights) {
		try {
			for (int i = 0; i < issues.size(); i++) {
				Evaluator ev = fEvaluators.get(issues.get(i));
				ev.setWeight(weights[i]); // set weight
			}
		} catch (NullPointerException npe) {
		}
	}

	/**
	 * Helper function, evaluator values need to be normalized
	 */
	public void normalizeWeights() {
		double sum = 0.0;
		for (Objective obj : fEvaluators.keySet()) {
			sum += fEvaluators.get(obj).getWeight();
		}

		for (Objective obj : fEvaluators.keySet()) {
			fEvaluators.get(obj)
					.setWeight(fEvaluators.get(obj).getWeight() / sum);
		}
	}

	/**
	 * @deprecated Use getObjective
	 * 
	 * @param index
	 *            The index of the issue to
	 * @return the indexed objective or issue
	 */
	@Deprecated
	public final Objective getIssue(int index) {
		return getDomain().getIssues().get(index);
	}

	/**
	 * Sets an [Objective, evaluator] pair. Replaces old evaluator for
	 * objective.
	 * 
	 * @param obj
	 *            The Objective to attach an Evaluator to.
	 * @param ev
	 *            The Evaluator to attach.
	 * @return The given evaluator.
	 */
	public final Evaluator addEvaluator(Objective obj, Evaluator ev) {
		fEvaluators.put(obj, ev);
		return ev;
	}

	/**
	 * @return The set with all pairs of evaluators and objectives in this
	 *         utility space.
	 */
	public final Set<Map.Entry<Objective, Evaluator>> getEvaluators() {
		return fEvaluators.entrySet();
	}

	/**
	 * Place a lock on the weight of an objective or issue. Mainly used to guide
	 * the normalization procedure.
	 * 
	 * @param obj
	 *            The objective or issue that is about to have it's weight
	 *            locked.
	 * @return <code>true</code> if successful, <code>false</code> If the
	 *         objective doesn't have an evaluator yet.
	 */
	public final boolean lock(Objective obj) {
		try {
			fEvaluators.get(obj).lockWeight();
		} catch (Exception e) {
			e.printStackTrace();
			return false;
		}
		return true;
	}

	/**
	 * Clear a lock on the weight of an objective or issue. Mainly used to guide
	 * the normalization procedure.
	 * 
	 * @param obj
	 *            The objective or issue that is having it's lock cleared.
	 * @return <code>true</code> If the lock is cleared, <code>false</code> if
	 *         the objective or issue doesn't have an evaluator yet.
	 */
	public final boolean unlock(Objective obj) {
		try {
			fEvaluators.get(obj).unlockWeight();
		} catch (Exception e) {
			e.printStackTrace();
			return false;
		}
		return true;
	}

	/**
	 * Normalizes the weights of non-locked objectives of the given objective so
	 * that all objective weights they sum up to one.
	 * 
	 * @param obj
	 *            of which the weights must be normalized.
	 * @return all evaluators using getEvaluators().
	 */
	public final Set<Map.Entry<Objective, Evaluator>> normalizeChildren(
			Objective obj) {
		Enumeration<Objective> childs = obj.children();
		double RENORMALCORR = 0.05;
		/*
		 * we add this to all weight sliders to solve the slider-stuck-at-0
		 * problem.
		 */
		double weightSum = 0;
		double lockedWeightSum = 0;
		int freeCount = 0;
		int lockedCount = 0;
		while (childs.hasMoreElements()) {
			Objective tmpObj = childs.nextElement();
			try {
				if (!fEvaluators.get(tmpObj).weightLocked()) {
					weightSum += fEvaluators.get(tmpObj).getWeight();
					freeCount++;
				} else {
					lockedWeightSum += fEvaluators.get(tmpObj).getWeight();
					lockedCount++;
				}
			} catch (Exception e) {

				// do nothing, we can encounter Objectives/issues without
				// Evaluators.
			}
		}
		if (freeCount + lockedCount == 1) {
			Enumeration<Objective> singleChild = obj.children();
			while (singleChild.hasMoreElements()) {
				Objective tmpObj = singleChild.nextElement();
				fEvaluators.get(tmpObj).setWeight(1.0);
			}
		}

		if (freeCount > 1) {
			Enumeration<Objective> normalChilds = obj.children();
			while (normalChilds.hasMoreElements()) {
				Objective tmpObj = normalChilds.nextElement();
				double diff = (lockedWeightSum + weightSum) - 1.0;
				// because of RENORMALCORR, total weight will get larger.
				double correctedWeightSum = weightSum
						+ RENORMALCORR * freeCount;
				try {

					if (!fEvaluators.get(tmpObj).weightLocked()) {
						double currentWeight = fEvaluators.get(tmpObj)
								.getWeight();
						double newWeight = currentWeight
								- (diff * (currentWeight + RENORMALCORR)
										/ correctedWeightSum);
						if (newWeight < 0) {
							newWeight = 0;
						}
						fEvaluators.get(tmpObj).setWeight(newWeight);
					}
				} catch (Exception e) {
					// do nothing, we can encounter Objectives/issues without
					// Evaluators.
				}

			}

		}

		return getEvaluators();
	}

	/****************** private internal stuff *******************/

	/**
	 * @param filename
	 *            The name of the xml file to parse.
	 * @throws IOException
	 *             if error occurs, e.g. file not found
	 */
	protected boolean loadTreeFromFile(String filename) throws IOException {
		SimpleDOMParser parser = new SimpleDOMParser();
		BufferedReader file = new BufferedReader(
				new FileReader(new File(filename)));
		SimpleElement root = parser.parse(file);
		return loadTreeRecursive(root);
	}

	/**
	 * Loads the weights and issues for the evaluators.
	 * 
	 * @param root
	 *            The current root of the XML structure.
	 * @return true iff error occured
	 */
	protected boolean loadTreeRecursive(SimpleElement currentRoot) {
		int nrOfWeights = 0;

		int index;
		// load reservation value
		try {
			if ((currentRoot.getChildByTagName(RESERVATION) != null)
					&& (currentRoot
							.getChildByTagName(RESERVATION).length > 0)) {
				SimpleElement xmlReservation = (SimpleElement) (currentRoot
						.getChildByTagName(RESERVATION)[0]);
				setReservationValue(
						Double.valueOf(xmlReservation.getAttribute("value")));
			}
		} catch (Exception e) {
			System.out.println("Utility space has no reservation value");
		}
		// load discount factor
		try {
			if ((currentRoot.getChildByTagName("discount_factor") != null)
					&& (currentRoot
							.getChildByTagName("discount_factor").length > 0)) {
				SimpleElement xmlReservation = (SimpleElement) (currentRoot
						.getChildByTagName("discount_factor")[0]);
				double df = Double
						.parseDouble(xmlReservation.getAttribute("value"));
				setDiscount(validateDiscount(df));
			}
		} catch (Exception e) {
			System.out.println("Utility space has no discount factor;");
		}

		Vector<Evaluator> tmpEvaluator = new Vector<Evaluator>();
		/*
		 * tmp vector with all Evaluators at this level. Used to normalize
		 * weigths.
		 */
		EVALUATORTYPE evalType;
		String type, etype;
		Evaluator lEvaluator = null;

		// Get the weights of the current children
		Object[] xml_weights = currentRoot.getChildByTagName("weight");
		nrOfWeights = xml_weights.length; // assuming each
		HashMap<Integer, Double> tmpWeights = new HashMap<Integer, Double>();
		for (int i = 0; i < nrOfWeights; i++) {
			index = Integer.valueOf(
					((SimpleElement) xml_weights[i]).getAttribute("index"));
			Double dval = Double.parseDouble(
					((SimpleElement) xml_weights[i]).getAttribute("value"));
			tmpWeights.put(index, dval);
		}

		// Collect evaluations for each of the issue values from file.
		// Assumption: Discrete-valued issues.
		Object[] xml_issues = currentRoot.getChildByTagName("issue");
		Object[] xml_objectives = currentRoot.getChildByTagName("objective");
		Object[] xml_obj_issues = new Object[xml_issues.length
				+ xml_objectives.length];
		int i_ind;
		for (i_ind = 0; i_ind < xml_issues.length; i_ind++) {
			xml_obj_issues[i_ind] = xml_issues[i_ind];
		}
		for (int o_ind = 0; (o_ind + i_ind) < xml_obj_issues.length; o_ind++) {
			xml_obj_issues[(o_ind + i_ind)] = xml_objectives[o_ind];
		}

		for (int i = 0; i < xml_obj_issues.length; i++) {
			index = Integer.valueOf(
					((SimpleElement) xml_obj_issues[i]).getAttribute("index"));
			type = ((SimpleElement) xml_obj_issues[i]).getAttribute("type");
			etype = ((SimpleElement) xml_obj_issues[i]).getAttribute("etype");
			if (type == null) { // No value type specified.
				evalType = EVALUATORTYPE.DISCRETE;
			} else if (type.equals(etype)) {
				evalType = EVALUATORTYPE.convertToType(type);
			} else if (type != null && etype == null) {
				evalType = EVALUATORTYPE.convertToType(type);
			} else {
				System.out.println(
						"Conflicting value types specified for evaluators in utility template file.");
				evalType = EVALUATORTYPE.convertToType(type);
			}
			if (tmpWeights.get(index) != null) {
				switch (evalType) {
				case DISCRETE:
					lEvaluator = new EvaluatorDiscrete();
					break;
				case INTEGER:
					lEvaluator = new EvaluatorInteger();
					break;
				case REAL:
					lEvaluator = new EvaluatorReal();
					break;
				case OBJECTIVE:
					lEvaluator = new EvaluatorObjective();
					break;
				}
				lEvaluator.loadFromXML((SimpleElement) (xml_obj_issues[i]));

				try {
					fEvaluators.put(
							getDomain().getObjectivesRoot().getObjective(index),
							lEvaluator);
					/*
					 * get the Objective or Issue.
					 */
				} catch (Exception e) {
					System.out.println("Domain-utilityspace mismatch");
					e.printStackTrace();
					return false;
				}
			}
			try {
				if (nrOfWeights != 0) {
					double tmpdwt = tmpWeights.get(index).doubleValue();
					Objective tmpob = getDomain().getObjectivesRoot()
							.getObjective(index);
					fEvaluators.get(tmpob).setWeight(tmpdwt);
				}
			} catch (Exception e) {
				System.out.println(
						"Evaluator-weight mismatch or no weight for this issue or objective.");
			}
			tmpEvaluator.add(lEvaluator); // for normalisation purposes.
		}

		// Recurse over all children:
		boolean returnval = false;
		Object[] objArray = currentRoot.getChildElements();
		for (int i = 0; i < objArray.length; i++)
			returnval = loadTreeRecursive((SimpleElement) objArray[i]);
		return returnval;
	}

	/**
	 * Adds the utilities (weights) from this utility space to a given domain.
	 * It modifies the currentLevel so the return value is superfluous.
	 * 
	 * @param currentLevel
	 *            is pointer to a XML tree describing the domain.
	 * @return XML tree with the weights. NOTE: currentLevel is modified anyway.
	 */
	private SimpleElement toXMLrecurse(SimpleElement currentLevel) {
		// go through all tags.

		Object[] Objectives = currentLevel.getChildByTagName("objective");

		for (int objInd = 0; objInd < Objectives.length; objInd++) {
			SimpleElement currentChild = (SimpleElement) Objectives[objInd];
			int childIndex = Integer
					.valueOf(currentChild.getAttribute("index"));
			try {
				Evaluator ev = fEvaluators.get(getDomain().getObjectivesRoot()
						.getObjective(childIndex));
				SimpleElement currentChildWeight = new SimpleElement("weight");
				currentChildWeight.setAttribute("index", "" + childIndex);
				currentChildWeight.setAttribute("value", "" + ev.getWeight());
				currentLevel.addChildElement(currentChildWeight);
			} catch (Exception e) {
				// do nothing, not every node has an evaluator.
			}
			currentChild = toXMLrecurse(currentChild);
		}

		Object[] Issues = currentLevel.getChildByTagName("issue");

		for (int issInd = 0; issInd < Issues.length; issInd++) {
			SimpleElement issueL = (SimpleElement) Issues[issInd];

			// set the weight
			int childIndex = Integer.valueOf(issueL.getAttribute("index"));
			Objective tmpEvObj = getDomain().getObjectivesRoot()
					.getObjective(childIndex);
			try {

				Evaluator ev = fEvaluators.get(tmpEvObj);

				SimpleElement currentChildWeight = new SimpleElement("weight");
				currentChildWeight.setAttribute("index", "" + childIndex);
				currentChildWeight.setAttribute("value", "" + ev.getWeight());
				currentLevel.addChildElement(currentChildWeight);

				String evtype_str = issueL.getAttribute("etype");
				EVALUATORTYPE evtype = EVALUATORTYPE.convertToType(evtype_str);
				switch (evtype) {
				case DISCRETE:
					// fill this issue with the relevant weights to items.
					Object[] items = issueL.getChildByTagName("item");
					for (int itemInd = 0; itemInd < items.length; itemInd++) {
						IssueDiscrete theIssue = (IssueDiscrete) getDomain()
								.getObjectivesRoot().getObjective(childIndex);

						EvaluatorDiscrete dev = (EvaluatorDiscrete) ev;
						Double eval = dev
								.getDoubleValue(theIssue.getValue(itemInd));
						((SimpleElement) items[itemInd])
								.setAttribute("evaluation", "" + eval);
					}
					break;
				case INTEGER:
					EvaluatorInteger iev = (EvaluatorInteger) ev;
					issueL.setAttribute("lowerbound", "" + iev.getLowerBound());
					issueL.setAttribute("upperbound", "" + iev.getUpperBound());

					SimpleElement thisIntEval = new SimpleElement("evaluator");
					thisIntEval.setAttribute("ftype", "linear");
					thisIntEval.setAttribute("slope", "" + iev.getSlope());
					thisIntEval.setAttribute("offset", "" + iev.getOffset());
					issueL.addChildElement(thisIntEval);
					break;
				case REAL:
					EvaluatorReal rev = (EvaluatorReal) ev;
					SimpleElement thisRealEval = new SimpleElement("evaluator");
					EVALFUNCTYPE revtype = rev.getFuncType();
					if (revtype == EVALFUNCTYPE.LINEAR) {
						thisRealEval.setAttribute("ftype", "linear");
						thisRealEval.setAttribute("parameter1",
								"" + rev.getLinearParam());
					} else if (revtype == EVALFUNCTYPE.CONSTANT) {
						thisRealEval.setAttribute("ftype", "constant");
						thisRealEval.setAttribute("parameter0",
								"" + rev.getConstantParam());
					}
					issueL.addChildElement(thisRealEval);
					break;
				}
			} catch (Exception e) {
				// do nothing, it could be that this objective/issue doesn't
				// have an evaluator yet.
			}

		}

		return currentLevel;
	}

	/**
	 * createFrom a default evaluator for a given Objective. This function is
	 * placed here, and not in Objective, because the Objectives should not be
	 * loaded with utility space functionality. The price we pay for that is
	 * that we now have an ugly switch inside the code, losing some modularity.
	 * 
	 * @param obj
	 *            the objective to createFrom an evaluator for
	 * @return the default evaluator
	 */
	private Evaluator defaultEvaluator(Objective obj) {
		if (obj.isObjective())
			return new EvaluatorObjective();
		// if not an objective then it must be an issue.
		switch (((Issue) obj).getType()) {
		case DISCRETE:
			return new EvaluatorDiscrete();
		case INTEGER:
			return new EvaluatorInteger();
		case REAL:
			return new EvaluatorReal();
		default:
			System.out.println("INTERNAL ERROR: issue of type "
					+ ((Issue) obj).getType() + "has no default evaluator");
		}
		return null;
	}

	/**
	 * Checks the normalization throughout the tree. Will eventually replace
	 * checkNormalization
	 * 
	 * @return true if the weigths are indeed normalized, false if they aren't.
	 */
	private boolean checkTreeNormalization() {
		return checkTreeNormalizationRecursive(getDomain().getObjectivesRoot());
	}

	/**
	 * Private helper function to check the normalisation throughout the tree.
	 * 
	 * @param currentRoot
	 *            The current parent node of the subtree we are going to check
	 * @return True if the weights are indeed normalized, false if they aren't.
	 */
	private boolean checkTreeNormalizationRecursive(Objective currentRoot) {
		boolean normalised = true;
		double lSum = 0;

		Enumeration<Objective> children = currentRoot.children();

		while (children.hasMoreElements() && normalised) {

			Objective tmpObj = children.nextElement();
			lSum += (fEvaluators.get(tmpObj)).getWeight();

		}
		return (normalised && lSum > .98 && lSum < 1.02);
	}

	public Map<Objective, Evaluator> getfEvaluators() {
		return fEvaluators;
	}
}
