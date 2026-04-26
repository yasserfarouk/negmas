package genius.core;

import java.util.List;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import java.io.Serializable;

import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlElementRef;
import javax.xml.bind.annotation.XmlRootElement;
import javax.xml.bind.annotation.adapters.XmlAdapter;
import javax.xml.bind.annotation.adapters.XmlJavaTypeAdapter;

import genius.core.analysis.pareto.IssueValue;
import genius.core.issue.Issue;
import genius.core.issue.Value;

/**
 * A bid is a set of tuples [idnumber,value], where idnumber is the unique
 * number of the issue, and value is the picked alternative.
 * <p>
 * Bid is a immutable. But you can create modified copies using
 * {@link #putValue(int, Value)}.
 * 
 * Bid should be considered final so do not extend this.
 * 
 * @author Dmytro Tykhonov, Koen Hindriks
 */
@XmlRootElement
public class Bid implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = -7723017380013100614L;

	private final Domain fDomain;

	/**
	 * the bid values for each IssueID
	 */
	@XmlElement(name = "values")
	@XmlJavaTypeAdapter(MyMapAdapter.class)
	private HashMap<Integer, Value> fValues;

	/**
	 * Only for (de)serialization
	 */
	private Bid() {
		fDomain = null; // keep Java happy. Serializer shall overwrite anyway
	}

	/**
	 * Create a new empty bid of which the values still must be set.
	 * 
	 * @param domain
	 *            the domain for this bid
	 */
	public Bid(Domain domain) {
		fDomain = domain;
		fValues = new HashMap<Integer, Value>();
	}

	/**
	 * createFrom a new bid in a domain. There is only this constructor because we require that ALL
	 * values in the domain get assigned a value.
	 * 
	 * @param domainP
	 *            the domain in which the bid is done
	 * @param bidP
	 *            HashMap, which is a set of pairs [issueID,value]
	 */
	public Bid(Domain domainP, HashMap<Integer, Value> bidP) {
		this.fDomain = domainP; 
		fValues = bidP;
	}

	/**
	 * create bid from set of {@link IssueValue}s
	 * 
	 * @param domain
	 *            the {@link Domain}
	 * @param values
	 *            a {@link Collection} of {@link IssueValue}s
	 */
	public Bid(Domain domain, Collection<IssueValue> values) {
		if (domain == null)
			throw new NullPointerException("null domain");
		if (values == null)
			throw new NullPointerException("null values");

		this.fDomain = domain;
		fValues = new HashMap<>();
		for (IssueValue iv : values) {
			fValues.put(iv.getIssue().getNumber(), iv.getValue());
		}
	}

	/**
	 * This method clones the given bid.
	 * 
	 * @param bid
	 *            the bid to clone
	 */
	public Bid(Bid bid) {

		fDomain = bid.fDomain;
		fValues = (HashMap<Integer, Value>) bid.fValues.clone();
	}

	/**
	 * @param issueNr
	 *            number of an issue.
	 * @return the picked value for given issue idnumber
	 * @throws IllegalArgumentException
	 *             if there exist no issue with the given number.
	 */
	public Value getValue(int issueNr) {
		Value v = fValues.get(issueNr);
		if (v == null) {
			if (fDomain.getIssues().get(issueNr) == null)
				throw new IllegalArgumentException("Bid.getValue: issue " + issueNr + " does not exist at all");
			throw new IllegalStateException("There is no evaluator for issue " + issueNr);
		}
		return v;
	}
	
	/**
	 * @param issue the issue
	 * @return the picked value for given issue
	 * @throws IllegalArgumentException
	 *             if there exist no issue with the given number.
	 */
	public Value getValue(Issue issue) 
	{
		return getValue(issue.getNumber());
	}
	
	/**
	 * @param issue the issue corresponding to the value. 
	 * This is needed because the same values can occur multiple times in a bid
	 * @param value
	 * @return Whether this bid has a value selected for an issue
	 */
	public boolean containsValue(Issue issue, Value value)
	{
		return getValue(issue).equals(value);
	}

	/**
	 * @param issueId
	 *            unique ID of an issue.
	 * @param pValue
	 *            value of the issue.
	 * @return new Bid as the current bid but with the value of the issue with
	 *         the given issueID to the given value
	 * @throws IllegalArgumentException
	 *             if there exist no issue with the given number.
	 */
	public Bid putValue(int issueId, Value pValue) {
		if (fValues.get(issueId).getType() != pValue.getType()) {
			// FIXME
			// if (fDomain.getIssue(issueId).getType() != pValue.getType()) {
			throw new IllegalArgumentException("expected value of type " + fDomain.getIssues().get(issueId).getType()
					+ " but got " + pValue + " of type " + pValue.getType());
		}
		HashMap<Integer, Value> newValues = new HashMap<Integer, Value>(fValues);
		newValues.put(issueId, pValue);
		return new Bid(fDomain, newValues);
	}
	

	public String toString() {
		String s = "Bid[";
		Set<Entry<Integer, Value>> value_set = fValues.entrySet();
		Iterator<Entry<Integer, Value>> value_it = value_set.iterator();
		int i = 0;
		while (value_it.hasNext()) {
			int ind = ((Entry<Integer, Value>) value_it.next()).getKey();
			Object tmpobj = fDomain.getObjectivesRoot().getObjective(ind);
			if (tmpobj != null) {
				String issueName = fDomain.getObjectivesRoot().getObjective(ind).getName();
				s += (i++ > 0 ? ", " : "") + issueName + ": " + fValues.get(ind);
			} else {
				System.out.println("objective with index " + ind + " does not exist");
			}
		}
		s = s + "]";
		return s;
	}
	
	/**
	 * @return A CSV version of the bid, useful for logging.
	 */
	public String toStringCSV() 
	{
		String s = "";
		Set<Entry<Integer, Value>> value_set = fValues.entrySet();
		Iterator<Entry<Integer, Value>> value_it = value_set.iterator();
		int i = 0;
		while (value_it.hasNext()) 
		{
			int ind = ((Entry<Integer, Value>) value_it.next()).getKey();
			s += (i++ > 0 ? ";" : "") + fValues.get(ind);
		}
		return s;
	}

	/**
	 * @param pBid
	 *            to which this bid must be compared.
	 * @return true if the values of this and the given bid are equal.
	 */
	public boolean equals(Bid pBid) {
		if (pBid == null)
			return false;
		return fValues.equals(pBid.fValues);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see java.lang.Object#equals(java.lang.Object)
	 */
	@Override
	public boolean equals(Object obj) {
		if (obj instanceof Bid)
			return equals((Bid) obj);
		return false;
	}

	/**
	 * @return a (copy of ) the list of all values in this bid. FIXME we really
	 *         should return an immutable {@link Map} here but that may break
	 *         many agents.
	 */

	public HashMap<Integer, Value> getValues() {
		return new HashMap<Integer, Value>(fValues);
	}

	// Reyhan: add this method
	public List<Issue> getIssues() {
		return fDomain.getIssues();
	}

	public Domain getDomain() {
		return fDomain;
	}
	
	/**
	 * Counts the number of equal values with another bid (assuming they are defined on the same domain)
	 */
	public int countEqualValues(Bid b)
	{
		int count = 0;
		for (Integer v : fValues.keySet())
		{
			if (this.fValues.get(v).equals(b.fValues.get(v)))
				count++;
		}
		return count;
	}
	
	/**
	 * Computes a basic distance function between 2 bids defined on the same domain:
	 * The average of their # of unequal values.
	 * @param b
	 * @return double in [0,1].
	 */
	
	public double getDistance(Bid b) {
		double nrOfIssues = this.getIssues().size();
		double unequalValues = nrOfIssues - this.countEqualValues(b);
		return unequalValues/nrOfIssues;	
	}
	

	@Override
	public int hashCode() {
		int code = 0;
		for (Entry<Integer, Value> lEntry : fValues.entrySet()) {
			code = code + lEntry.getValue().hashCode();
		}
		return code;// fValues.hashCode();
	}

}

class MyMapAdapter extends XmlAdapter<Temp, Map<Integer, Value>> {

	@Override
	public Temp marshal(Map<Integer, Value> arg0) throws Exception {
		Temp temp = new Temp();
		for (Entry<Integer, Value> entry : arg0.entrySet()) {
			temp.entry.add(new Item(entry.getKey(), entry.getValue()));
		}
		return temp;
	}

	@Override
	public Map<Integer, Value> unmarshal(Temp arg0) throws Exception {
		Map<Integer, Value> map = new HashMap<Integer, Value>();
		for (Item item : arg0.entry) {
			map.put(item.key, item.value);
		}
		return map;
	}

}

class Temp {
	@XmlElement(name = "issue")
	public List<Item> entry;

	public Temp() {
		entry = new ArrayList<Item>();
	}

}

@XmlRootElement
class Item {
	@XmlAttribute(name = "index")
	public Integer key;

	@XmlElementRef
	public Value value;

	public Item() {
	}

	public Item(Integer key, Value val) {
		this.key = key;
		this.value = val;
	}
}
