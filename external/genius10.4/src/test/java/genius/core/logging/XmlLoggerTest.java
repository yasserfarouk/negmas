package genius.core.logging;

import static org.junit.Assert.assertEquals;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import java.io.FileNotFoundException;
import java.io.OutputStream;
import java.util.LinkedHashMap;
import java.util.Map;

import javax.xml.stream.XMLStreamException;

import org.junit.Before;
import org.junit.Test;

import genius.core.events.AgentLogEvent;

public class XmlLoggerTest {

	private static final String AGENT1 = "Agent1";
	private XmlLogger logger;

	@Before
	public void before() throws FileNotFoundException, XMLStreamException {
		logger = new XmlLogger(mock(OutputStream.class), "test");
	}

	@Test
	public void testAgentLogEventSmokeTest() {
		AgentLogEvent logEvent = mock(AgentLogEvent.class);
		when(logEvent.getAgent()).thenReturn(AGENT1);
		when(logEvent.getLog()).thenReturn(agentLog(AGENT1));

		logger.notifyChange(logEvent);

	}

	@Test
	public void testAgentLogEvent() {
		AgentLogEvent logEvent = mock(AgentLogEvent.class);
		when(logEvent.getAgent()).thenReturn(AGENT1);
		when(logEvent.getLog()).thenReturn(agentLog(AGENT1));

		logger.notifyChange(logEvent);
		if (!(agentLog(AGENT1).equals(logger.agentLogs.get(AGENT1)))) {
			System.out.println("NOT EQUAL!");
		}
		assertEquals((Map) agentLog(AGENT1), (Map) logger.agentLogs.get(AGENT1));

		agentLog("a").equals(agentLog("b"));
	}

	/**
	 * @param name
	 * @return
	 */
	private Map<String, String> agentLog(String name) {
		Map<String, String> outcome = new LinkedHashMap<>();
		outcome.put("agent", name);
		outcome.put("agentName", "QAgent");
		return outcome;
	}

	/**
	 * {@link Map#equals(Object)} does not work if the
	 * 
	 * @param map1
	 * @param map2
	 */
	void checkEqual(Map map1, Map map2) {
		assertEquals(map1.keySet().size(), map2.keySet().size());
	}

}
