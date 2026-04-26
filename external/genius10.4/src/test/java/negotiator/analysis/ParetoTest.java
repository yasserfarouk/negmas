package negotiator.analysis;

import static org.junit.Assert.assertNull;

import java.io.IOException;
import java.lang.management.ManagementFactory;
import java.lang.management.ThreadMXBean;
import java.util.ArrayList;
import java.util.List;

import org.junit.Before;
import org.junit.Test;
import org.xml.sax.Attributes;
import org.xml.sax.SAXException;
import org.xml.sax.XMLReader;
import org.xml.sax.helpers.DefaultHandler;
import org.xml.sax.helpers.XMLReaderFactory;

import genius.core.Domain;
import genius.core.DomainImpl;
import genius.core.analysis.BidPoint;
import genius.core.analysis.BidSpace;
import genius.core.analysis.pareto.ParetoFrontierF;
import genius.core.analysis.pareto.PartialBidPoint;
import genius.core.bidding.BidDetails;
import genius.core.boaframework.SortedOutcomeSpace;
import genius.core.misc.Range;
import genius.core.qualitymeasures.ScenarioInfo;
import genius.core.utility.AdditiveUtilitySpace;
import genius.domains.DomainInstaller;

/**
 * This class can be used to test if the implementation of the Pareto frontier
 * algorithm in BidSpace returns the correct results on each domain. The
 * efficient algorithm is compared against a simple bruteforce algorithm.
 * 
 * No effort was made to optimize the bruteforce algorithm as I wanted to be
 * sure that it is correct. Therefore, it is not advised to check domains with
 * more than 200.000 bids.
 * 
 * @author Mark Hendrikx
 */
public class ParetoTest {

	private static final double EPSILON = 0.000000001;

	@Before
	public void before() throws IOException {
		DomainInstaller.run();
	}

	@Test
	public void testPareto() throws Exception {
		process();
	}

	class SimpleTimer {
		ThreadMXBean bean = ManagementFactory.getThreadMXBean();
		long start = bean.getCurrentThreadUserTime();
		long end;

		public void stop() {
			end = bean.getCurrentThreadUserTime();
		}

		public double time() {
			return (end - start) / 1000000000.;
		}
	}

	/**
	 * Simple method to compare if the algorithm for calculating the Pareto-bids
	 * in the BidSpace class returns the right results.
	 * 
	 * @param dir
	 *            in which Genius is installed
	 * @throws Exception
	 *             when an error occurs on parsing the files.
	 */
	public void process() throws Exception {
		ArrayList<ScenarioInfo> domains = parseDomainFile();
		ThreadMXBean bean = ManagementFactory.getThreadMXBean();

		for (ScenarioInfo domainSt : domains) {
			System.out.println("testing pareto on domain " + domainSt);
			// 1. Load the domain
			Domain domain = new DomainImpl(domainSt.getDomain());
			AdditiveUtilitySpace utilitySpaceA, utilitySpaceB;
			utilitySpaceA = new AdditiveUtilitySpace(domain,
					domainSt.getPrefProfA());
			utilitySpaceB = new AdditiveUtilitySpace(domain,
					domainSt.getPrefProfB());
			System.out.println("Timing for " + domain.getName() + "("
					+ domain.getNumberOfPossibleBids() + "):");

			List<BidPoint> realParetoBids = null;

			// 2. Determine all Pareto-bids with various algorithms

			SimpleTimer bruteForceT = new SimpleTimer();
			if (domain.getNumberOfPossibleBids() < 10000) {
				realParetoBids = bruteforceParetoBids(domain, utilitySpaceA,
						utilitySpaceB);
			}
			bruteForceT.stop();

			SimpleTimer bidSpaceT = new SimpleTimer();
			BidSpace space = new BidSpace(utilitySpaceA, utilitySpaceB, true);
			List<BidPoint> estimatedParetoBids = space.getParetoFrontier();
			bidSpaceT.stop();

			SimpleTimer fastT = new SimpleTimer();
			List<BidPoint> fastPareto = doFastPareto(utilitySpaceA,
					utilitySpaceB);
			fastT.stop();

			// 3. Check if there is a difference in the output
			if (realParetoBids != null) {
				assertNull(
						"Problem in estimatedPareto with domain "
								+ domain.getName(),
						checkValidity(estimatedParetoBids, realParetoBids));
			}

			assertNull("Problem in fastPareto with domain " + domain.getName(),
					checkValidity(fastPareto, estimatedParetoBids));

			System.out.println("bruteforce search:" + (realParetoBids == null
					? "skipped" : bruteForceT.time()));
			System.out.println("bidSpace search:" + bidSpaceT.time());
			System.out.println("fast search:" + fastT.time());
		}
		System.out.println("Finished processing domains");
	}

	/**
	 * Does the heavy plumbing job
	 * 
	 * @throws Exception
	 * 
	 */
	private List<BidPoint> doFastPareto(AdditiveUtilitySpace utilitySpaceA,
			AdditiveUtilitySpace utilitySpaceB) throws Exception {

		ParetoFrontierF fastparetof = new ParetoFrontierF(utilitySpaceA,
				utilitySpaceB);

		ArrayList<BidPoint> bidpoints = new ArrayList<BidPoint>();
		for (PartialBidPoint point : fastparetof.getFrontier()) {
			bidpoints.add(new BidPoint(null, point.utilA(), point.utilB()));
		}
		return bidpoints;
	}

	/**
	 * Check if the output of the efficient algorithm and the brutefore
	 * algorithm to calculate the Pareto-optimal bids are identical.
	 * 
	 * @param estimatedParetoBids
	 *            Pareto-bids as estimated by an efficient algorithm in the
	 *            BidSpace class.
	 * @param realParetoBids
	 *            Pareto-bids as calculated by the bruteforce algorithm.
	 * @return null if sets are equal, or non-null string describing difference.
	 */
	private static String checkValidity(List<BidPoint> set1,
			List<BidPoint> set2) {
		if (set1.size() != set2.size()) {
			return "pareto sets are not equal size: " + set1.size() + " versus "
					+ set2.size();
		}
		for (BidPoint paretoBid : set1) {
			boolean found = false;
			for (int a = 0; a < set2.size(); a++) {
				if (Math.abs(set2.get(a).getUtilityA()
						- paretoBid.getUtilityA()) < EPSILON
						&& Math.abs(set2.get(a).getUtilityB()
								- paretoBid.getUtilityB()) < EPSILON) {
					found = true;
					break;
				}
			}
			if (!found) {
				return "set 2 does not contain pareto bid " + paretoBid;
			}
		}

		return null;
	}

	/**
	 * Parses the domainrepository and returns a set of domain-objects
	 * containing all information.
	 * 
	 * @param dir
	 * @return set of domain-objects
	 * @throws Exception
	 */
	private static ArrayList<ScenarioInfo> parseDomainFile() throws Exception {
		XMLReader xr = XMLReaderFactory.createXMLReader();
		DomainParser handler = new DomainParser();
		xr.setContentHandler(handler);
		xr.setErrorHandler(handler);
		xr.parse("paretotestdomainrepository.xml");

		return handler.getDomains();
	}

	/**
	 * Bruteforce algorithm to calculate the Pareto-bids.
	 * 
	 * @param domain
	 * @param spaceA
	 * @param spaceB
	 * @return set of Pareto-bids
	 */
	private static ArrayList<BidPoint> bruteforceParetoBids(Domain domain,
			AdditiveUtilitySpace spaceA, AdditiveUtilitySpace spaceB) {
		SortedOutcomeSpace outcomeSpaceA = new SortedOutcomeSpace(spaceA);
		ArrayList<BidPoint> paretoBids = new ArrayList<BidPoint>();
		try {
			for (BidDetails bid : outcomeSpaceA.getAllOutcomes()) {
				double utilA = spaceA.getUtility(bid.getBid());
				double utilB = spaceB.getUtility(bid.getBid());
				boolean found = false;

				for (BidDetails otherBid : outcomeSpaceA
						.getBidsinRange(new Range(utilA - 0.01, 1.1))) { // -0.01
																			// as
																			// we
																			// want
																			// to
																			// include
																			// duplicates
					if ((otherBid != bid
							&& ((spaceA.getUtility(otherBid.getBid()) > utilA
									&& spaceB.getUtility(
											otherBid.getBid()) >= utilB))
							|| (otherBid != bid
									&& spaceA.getUtility(
											otherBid.getBid()) >= utilA
									&& spaceB.getUtility(
											otherBid.getBid()) > utilB))) {
						found = true;
						break;
					}
				}
				if (!found) {
					paretoBids.add(new BidPoint(bid.getBid(),
							bid.getMyUndiscountedUtil(),
							spaceB.getUtility(bid.getBid())));
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		return paretoBids;
	}

	/**
	 * Create an XML parser to parse the domainrepository.
	 */
	static class DomainParser extends DefaultHandler {

		ScenarioInfo domain = null;
		ArrayList<ScenarioInfo> domains = new ArrayList<ScenarioInfo>();

		@Override
		public void startElement(String nsURI, String strippedName,
				String tagName, Attributes attributes) throws SAXException {
			if (tagName.equals("domainRepItem") && attributes.getLength() > 0) {
				domain = new ScenarioInfo(
						attributes.getValue("url").substring(5));
			} else if (tagName.equals("profile")) {
				if (domain.getPrefProfA() == null) {
					domain.setPrefProfA(
							attributes.getValue("url").substring(5));
				} else if (domain.getPrefProfB() == null) {
					domain.setPrefProfB(
							attributes.getValue("url").substring(5));
				} else {
					System.out.println(
							"WARNING: Violation of two preference profiles per domain assumption for "
									+ strippedName);
				}
			}

		}

		@Override
		public void endElement(String nsURI, String strippedName,
				String tagName) throws SAXException {
			// domain is not null check is required, as the domainRepItem is
			// used in multiple contexts
			if (tagName.equals("domainRepItem") && domain != null) {
				domains.add(domain);
				domain = null;
			}
		}

		public ArrayList<ScenarioInfo> getDomains() {
			return domains;
		}
	}

}