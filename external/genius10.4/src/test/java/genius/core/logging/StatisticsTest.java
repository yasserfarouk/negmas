package genius.core.logging;

import static org.junit.Assert.assertEquals;

import javax.xml.bind.JAXBContext;
import javax.xml.bind.JAXBException;
import javax.xml.bind.Marshaller;

import org.junit.Test;

public class StatisticsTest {
	private final String agentname = "testagent";

	@Test
	public void testSerializer() throws JAXBException {
		AgentStatistics stat = new AgentStatistics(agentname, 12d, 16d, 1d, 1d, 1d, 33l);

		JAXBContext jc = JAXBContext.newInstance(AgentStatistics.class);
		Marshaller marshaller = jc.createMarshaller();
		marshaller.setProperty(Marshaller.JAXB_FORMATTED_OUTPUT, true);
		marshaller.marshal(stat, System.out);

	}

	@Test
	public void testMeans() throws JAXBException {
		AgentStatistics stat = new AgentStatistics(agentname, 12d, 16d, 1d, 2d, 3d, 33l);

		assertEquals(agentname, stat.getName());
		assertEquals(stat.getMeanUndiscounted(), 12d / 33l, .000001);
		assertEquals(stat.getMeanDiscounted(), 16d / 33l, .000001);
		assertEquals(stat.getMeanNashDistance(), 1d / 33l, .000001);
		assertEquals(stat.getMeanWelfare(), 2d / 33l, .000001);
		assertEquals(stat.getMeanParetoDistance(), 3d / 33l, .000001);

	}

	@Test
	public void testUpdate() {
		AgentStatistics stat = new AgentStatistics(agentname, 1d, 2d, 1d, 2d, 3d, 4l);
		stat = stat.withUtility(1d, 1d, 1d, 1d, 1d);

		assertEquals(agentname, stat.getName());
		assertEquals(2d / 5l, stat.getMeanUndiscounted(), .000001);
		assertEquals(3d / 5l, stat.getMeanDiscounted(), .000001);
		assertEquals(stat.getMeanNashDistance(), 2d / 5l, .000001);
		assertEquals(stat.getMeanWelfare(), 3d / 5l, .000001);
		assertEquals(stat.getMeanParetoDistance(), 4d / 5l, .000001);

	}

}
