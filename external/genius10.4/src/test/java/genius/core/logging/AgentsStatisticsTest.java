package genius.core.logging;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.io.UnsupportedEncodingException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

import javax.xml.bind.JAXBContext;
import javax.xml.bind.JAXBException;
import javax.xml.bind.Marshaller;
import javax.xml.bind.Unmarshaller;

import org.junit.Test;

public class AgentsStatisticsTest {
	private final String agentname = "testagent";
	private final String teststream = "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?> "
			+ "<agentsStatistics>    <statistics>   " + "    <agentname>testagent</agentname>     "
			+ " <totalUndiscountedUtility>22.0</totalUndiscountedUtility>        "
			+ " <totalDiscountedUtility>11.0</totalDiscountedUtility>      "
			+ " <numberOfSessions>33</numberOfSessions>   "
			+ " <meanDiscounted>0.3333333333333333</meanDiscounted>        "
			+ " <meanUndiscounted>0.6666666666666666</meanUndiscounted>" + "</statistics> </agentsStatistics>";

	@Test
	public void testGet() {
		List<AgentStatistics> statslist = new ArrayList<>();
		statslist.add(new AgentStatistics(agentname, 22, 11, 1, 1, 1, 33));
		AgentsStatistics stats = new AgentsStatistics(statslist);

		testStats(stats);
	}

	@Test
	public void testSerialize() throws JAXBException {
		List<AgentStatistics> statslist = new ArrayList<>();
		statslist.add(new AgentStatistics(agentname, 22, 11, 1, 1, 1, 33));
		AgentsStatistics stats = new AgentsStatistics(statslist);

		JAXBContext jc = JAXBContext.newInstance(AgentsStatistics.class);
		Marshaller marshaller = jc.createMarshaller();
		marshaller.setProperty(Marshaller.JAXB_FORMATTED_OUTPUT, true);
		marshaller.marshal(stats, System.out);

	}

	@Test
	public void testDeserialize() throws JAXBException, UnsupportedEncodingException {

		JAXBContext jc = JAXBContext.newInstance(AgentsStatistics.class);
		Unmarshaller unmarshaller = jc.createUnmarshaller();

		InputStream stream = new ByteArrayInputStream(teststream.getBytes(StandardCharsets.UTF_8.name()));

		AgentsStatistics stats = (AgentsStatistics) unmarshaller.unmarshal(stream);

	}

	@Test
	public void testAddNewAgent() {
		AgentsStatistics stats = new AgentsStatistics(new ArrayList<AgentStatistics>());
		stats = stats.withStatistics(agentname, 0.1d, 0.5d, 1, 1, 1);
		testStats(stats);
	}

	@Test
	public void testWithUtility() {
		List<AgentStatistics> statslist = new ArrayList<>();
		statslist.add(new AgentStatistics(agentname, 3, 4, 1, 1, 1, 5));
		AgentsStatistics stats = new AgentsStatistics(statslist);

		stats = stats.withStatistics(agentname, 1, 2, 1, 1, 1);
		AgentStatistics stat = stats.get(agentname);
		assertEquals(4. / 6., stat.getMeanUndiscounted(), 0.000001);
	}

	/**
	 * Internal test to see if we have indeed a
	 * 
	 * @param stats
	 */
	private void testStats(AgentsStatistics stats) {
		AgentStatistics stat = stats.get(agentname);
		assertNotNull("AgentStatistics should contain statistics for agent " + agentname, stat);
		assertEquals(agentname, stat.getName());
	}
}
