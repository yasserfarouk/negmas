package genius.core;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

import genius.core.issue.ISSUETYPE;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.IssueInteger;
import genius.core.issue.IssueReal;
import genius.core.issue.Objective;
import genius.core.issue.Value;
import genius.core.issue.ValueInteger;
import genius.core.issue.ValueReal;
import genius.core.xml.SimpleDOMParser;
import genius.core.xml.SimpleElement;

/**
 * Implementation of Domain. Representation of the outcome space of a scenario.
 *
 * @author Dmytro Tykhonov & Koen Hindriks
 */
public class DomainImpl implements Domain, Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = -8729366996052137300L;
	private Objective fObjectivesRoot = new Objective();
	private String name = "";
	private SimpleElement root;

	/**
	 * Creates an empty domain.
	 */
	public DomainImpl() {
	}

	/**
	 * @return XML-representation of this domain.
	 */
	public SimpleElement getXMLRoot() {
		return root;
	}

	/**
	 * Creates a domain given an XML-representation of the domain.
	 * 
	 * @param root
	 *            XML-representation of the domain.
	 */
	public DomainImpl(SimpleElement root) {
		this.root = root;
		loadTreeFromXML(root);
	}

	/**
	 * Creates a domain given the path to a file with an XML-representation.
	 * 
	 * @param filename
	 * @throws Exception
	 */
	public DomainImpl(String filename) throws IOException {
		this(new File(filename));
		name = filename;
	}

	/**
	 * read a domain from a file.
	 * 
	 * @param domainFile
	 *            file containing the domain
	 * @throws IOException
	 * @throws Exception
	 *             if file not found or containing unreadable (non xml) data.
	 */
	public DomainImpl(File domainFile) throws IOException {
		if (!domainFile.exists()) {
			throw new FileNotFoundException("File does not exist " + domainFile);
		}
		name = domainFile.getAbsolutePath();
		SimpleDOMParser parser = new SimpleDOMParser();
		BufferedReader file = new BufferedReader(new FileReader(domainFile));
		root = parser.parse(file);

		SimpleElement xml_utility_space;
		try {
			xml_utility_space = (SimpleElement) (root.getChildByTagName("utility_space")[0]);
		} catch (Exception err) {
			throw new IOException("Can't read from " + domainFile + ", incorrect format of file");
		}
		loadTreeFromXML(xml_utility_space);
	}

	@Override
	public final Objective getObjectivesRoot() {
		return fObjectivesRoot;
	}

	/**
	 * Sets a new domain root.
	 * 
	 * @param ob
	 *            The new root Objective
	 */
	public final void setObjectivesRoot(Objective ob) {
		fObjectivesRoot = ob;
	}

	/**
	 * @param pRoot
	 *            The SimpleElement that contains the root of the Objective
	 *            tree.
	 */
	private final void loadTreeFromXML(SimpleElement pRoot) {
		// SimpleElement root contains a LinkedList with SimpleElements.
		/*
		 * Structure of the file:
		 * 
		 * pRoot contains information about how many items there exist in the
		 * utilityspace. The first SimpleElement under pRoot contains the root
		 * objective of the tree, with a number of objective as tagnames.
		 */

		// Get the number of issues:

		// Get the actual root Objective.
		SimpleElement root = (SimpleElement) (pRoot.getChildByTagName("objective")[0]);
		int rootIndex = Integer.valueOf(root.getAttribute("index"));
		Objective objAlmostRoot = new Objective();
		objAlmostRoot.setNumber(rootIndex);
		String name = root.getAttribute("name");
		if (name != null)
			objAlmostRoot.setName(name);
		else
			objAlmostRoot.setName("root"); // just in case.
		// set objAlmostRoot attributes based on pRoot

		fObjectivesRoot = buildTreeRecursive(root, objAlmostRoot);

	}

	// added by Herbert
	/**
	 * 
	 * @param currentLevelRoot
	 *            The current SimpleElement containing the information for the
	 *            Objective on this level.
	 * @param currentParent
	 *            parent of the current level of this branch of the tree.
	 * @return The current parent of this level of the tree, with the children
	 *         attached.
	 */

	private final Objective buildTreeRecursive(SimpleElement currentLevelRoot, Objective currentParent) {
		Object[] currentLevelObjectives = currentLevelRoot.getChildByTagName("objective");
		Object[] currentLevelIssues = currentLevelRoot.getChildByTagName("issue");
		for (int i = 0; i < currentLevelObjectives.length; i++) {
			SimpleElement childObjectives = (SimpleElement) currentLevelObjectives[i];
			int obj_index = Integer.valueOf(childObjectives.getAttribute("index"));
			Objective child = new Objective(currentParent);
			child.setNumber(obj_index);
			// Set child attributes based on childObjectives.
			child.setName(childObjectives.getAttribute("name"));

			currentParent.addChild(buildTreeRecursive(childObjectives, child));

		}

		for (int j = 0; j < currentLevelIssues.length; j++) {
			Issue child = null;

			SimpleElement childIssues = (SimpleElement) currentLevelIssues[j];
			// check type of issue
			String name = childIssues.getAttribute("name");
			int index = Integer.parseInt(childIssues.getAttribute("index"));

			// Collect issue value type from XML file.
			String type = childIssues.getAttribute("type");
			String vtype = childIssues.getAttribute("vtype");
			ISSUETYPE issueType;
			if (type == null) {
				issueType = ISSUETYPE.DISCRETE;
			} else if (type.equals(vtype)) {
				// Both "type" as well as "vtype" attribute, but consistent.
				issueType = ISSUETYPE.convertToType(type);
			} else if (type != null && vtype == null) { // Used label "type"
														// instead of label
														// "vtype".
				issueType = ISSUETYPE.convertToType(type);
			} else {
				System.out.println("Conflicting value types specified for issue in template file.");
				// TODO: Define exception.
				// For now: use "type" label.
				issueType = ISSUETYPE.convertToType(type);
			}

			// Collect values and/or corresponding parameters for issue type.
			Object[] xml_items;
			Object[] xml_item;
			int nrOfItems, minI, maxI;
			double minR, maxR;
			String[] values;
			String[] desc;
			switch (issueType) {
			case DISCRETE:
				// Collect discrete values for discrete-valued issue from xml
				// template
				xml_items = childIssues.getChildByTagName("item");
				nrOfItems = xml_items.length;

				values = new String[nrOfItems];
				desc = new String[nrOfItems];
				for (int k = 0; k < nrOfItems; k++) {
					// TODO: check range of indexes.
					values[k] = ((SimpleElement) xml_items[k]).getAttribute("value");
					desc[k] = ((SimpleElement) xml_items[k]).getAttribute("description");
				}
				child = new IssueDiscrete(name, index, values, desc, currentParent);
				break;
			case INTEGER:
				// Collect range bounds for integer-valued issue from xml
				// template
				xml_item = childIssues.getChildByTagName("range");
				minI = Integer.valueOf(childIssues.getAttribute("lowerbound"));
				maxI = Integer.valueOf(childIssues.getAttribute("upperbound"));
				child = new IssueInteger(name, index, minI, maxI, currentParent);
				break;
			case REAL:
				// Collect range bounds for integer-valued issue from xml
				// template
				xml_item = childIssues.getChildByTagName("range");
				minR = Double.valueOf(((SimpleElement) xml_item[0]).getAttribute("lowerbound"));
				maxR = Double.valueOf(((SimpleElement) xml_item[0]).getAttribute("upperbound"));
				child = new IssueReal(name, index, minR, maxR);
				break;
			default: // By default, createFrom discrete-valued issue
				// Collect discrete values for discrete-valued issue from xml
				// template
				xml_items = childIssues.getChildByTagName("item");
				nrOfItems = xml_items.length;
				values = new String[nrOfItems];
				child = new IssueDiscrete(name, index, values, currentParent);
				break;
			}

			// Descriptions?
			child.setNumber(index);
			try {
				currentParent.addChild(child);
			} catch (Exception e) {
				System.out.println("child is NULL");
				e.printStackTrace();

			}
		}

		return currentParent;
	}

	@Override
	public Bid getRandomBid(Random r) {
		if (r == null)
			r = new Random();
		HashMap<Integer, Value> values = new HashMap<Integer, Value>();

		int lNrOfOptions, lOptionIndex;

		// For each issue, compute a random value to return in bid.
		for (Issue lIssue : getIssues()) {
			switch (lIssue.getType()) {
			case DISCRETE:
				IssueDiscrete lIssueDiscrete = (IssueDiscrete) lIssue;
				lNrOfOptions = lIssueDiscrete.getNumberOfValues();
				lOptionIndex = Double.valueOf(r.nextDouble() * (lNrOfOptions)).intValue();
				if (lOptionIndex >= lNrOfOptions)
					lOptionIndex = lNrOfOptions - 1;
				values.put(lIssue.getNumber(), lIssueDiscrete.getValue(lOptionIndex));
				break;
			case INTEGER:
				lNrOfOptions = ((IssueInteger) lIssue).getUpperBound() - ((IssueInteger) lIssue).getLowerBound() + 1;
				lOptionIndex = Double.valueOf(r.nextDouble() * (lNrOfOptions)).intValue();
				if (lOptionIndex >= lNrOfOptions)
					lOptionIndex = lNrOfOptions - 1;
				values.put(lIssue.getNumber(),
						new ValueInteger(((IssueInteger) lIssue).getLowerBound() + lOptionIndex));
				break;
			case REAL:
				IssueReal lIssueReal = (IssueReal) lIssue;
				lNrOfOptions = lIssueReal.getNumberOfDiscretizationSteps();
				double lOneStep = (lIssueReal.getUpperBound() - lIssueReal.getLowerBound()) / lNrOfOptions;
				lOptionIndex = Double.valueOf(r.nextDouble() * (lNrOfOptions)).intValue();
				if (lOptionIndex >= lNrOfOptions)
					lOptionIndex = lNrOfOptions - 1;
				values.put(lIssue.getNumber(), new ValueReal(lIssueReal.getLowerBound() + lOneStep * lOptionIndex));
				break;
			}
		}
		try {
			return new Bid(this, values);
		} catch (Exception e) {
			System.out.println("problem getrandombid:" + e.getMessage());
		}
		return null;
	}

	/**
	 * Creates an XML representation of this domain.
	 * 
	 * @return the SimpleElements representation of this Domain or
	 *         <code>null</code> when there was an error.
	 */
	public SimpleElement toXML() {
		SimpleElement root = new SimpleElement("negotiation_template");
		SimpleElement utilRoot = new SimpleElement("utility_space");
		// set attributes for this domain
		utilRoot.setAttribute("number_of_issues", "" + fObjectivesRoot.getChildCount());
		utilRoot.addChildElement(fObjectivesRoot.toXML());
		root.addChildElement(utilRoot);
		return root;
	}

	@Override
	public List<Objective> getObjectives() {
		Enumeration<Objective> objectives = fObjectivesRoot.getPreorderEnumeration();
		ArrayList<Objective> objectivelist = new ArrayList<Objective>();
		while (objectives.hasMoreElements())
			objectivelist.add(objectives.nextElement());
		return objectivelist;
	}

	@Override
	public List<Issue> getIssues() {
		Enumeration<Objective> issues = fObjectivesRoot.getPreorderIssueEnumeration();
		ArrayList<Issue> issuelist = new ArrayList<Issue>();
		while (issues.hasMoreElements())
			issuelist.add((Issue) issues.nextElement());
		return issuelist;
	}

	@Override
	public long getNumberOfPossibleBids() {
		List<Issue> lIssues = getIssues();
		if (lIssues.isEmpty()) {
			return 0;
		}
		long lNumberOfPossibleBids = (long) 1;
		for (Issue lIssue : lIssues) {
			switch (lIssue.getType()) {
			case DISCRETE:
				lNumberOfPossibleBids = lNumberOfPossibleBids * ((IssueDiscrete) lIssue).getNumberOfValues();
				break;
			case REAL:
				lNumberOfPossibleBids = lNumberOfPossibleBids * ((IssueReal) lIssue).getNumberOfDiscretizationSteps();
				break;
			case INTEGER:
				lNumberOfPossibleBids = lNumberOfPossibleBids
						* ((IssueInteger) lIssue).getNumberOfDiscretizationSteps();
				break;
			default:
				System.out.println("Unsupported type " + lIssue.getType());
				break;

			}
		}
		return lNumberOfPossibleBids;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((fObjectivesRoot == null) ? 0 : fObjectivesRoot.hashCode());
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		DomainImpl other = (DomainImpl) obj;
		if (fObjectivesRoot == null) {
			if (other.fObjectivesRoot != null)
				return false;
		} else if (!fObjectivesRoot.equals(other.fObjectivesRoot))
			return false;
		return true;
	}

	@Override
	public String getName() {
		return name;
	}
}
