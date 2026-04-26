package negotiator.repository;

import static org.junit.Assert.assertEquals;

import java.io.StringReader;
import java.io.StringWriter;

import javax.xml.bind.JAXBContext;
import javax.xml.bind.JAXBException;
import javax.xml.bind.Marshaller;
import javax.xml.bind.Unmarshaller;

import org.junit.Test;

import genius.core.boaframework.AcceptanceStrategy;
import genius.core.repository.boa.BoaRepItem;

public class AcceptanceConditionRepItemTest {

	@Test
	public void testDeserialize() throws JAXBException {
		JAXBContext jaxbContext = JAXBContext.newInstance(BoaRepItem.class);
		Unmarshaller unmarshaller = jaxbContext.createUnmarshaller();

		String TEXT = "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\n"
				+ "<boa classpath=\"class.path\"/>";
		StringReader reader = new StringReader(TEXT);

		@SuppressWarnings("unchecked")
		BoaRepItem<AcceptanceStrategy> element = (BoaRepItem<AcceptanceStrategy>) unmarshaller.unmarshal(reader);

		System.out.println(element);
		assertEquals("class.path", element.getClassPath());
		// The path does not exist, so this will throw.
		// assertEquals("name", element.getName());
	}

	@Test
	public void testSerialize() throws JAXBException {
		BoaRepItem<AcceptanceStrategy> ac = new BoaRepItem<>("class.path");
		StringWriter writer = new StringWriter();

		JAXBContext context = JAXBContext.newInstance(BoaRepItem.class);
		Marshaller m = context.createMarshaller();
		m.marshal(ac, writer);
		System.out.println(writer.toString());
	}

	@Test
	public void testDeserializeBoa() throws JAXBException {
		JAXBContext jaxbContext = JAXBContext.newInstance(BoaRepItem.class);
		Unmarshaller unmarshaller = jaxbContext.createUnmarshaller();

		String TEXT = "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?><boa classpath=\"class.path\"/>";
		StringReader reader = new StringReader(TEXT);

		BoaRepItem<AcceptanceStrategy> element = (BoaRepItem<AcceptanceStrategy>) unmarshaller.unmarshal(reader);

		System.out.println(element);
		assertEquals("class.path", element.getClassPath());
		// The path does not exist, so this will throw.
		// assertEquals("name", element.getName());
	}

}
