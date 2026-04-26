package negotiator.repository;

import static org.junit.Assert.assertEquals;

import java.io.StringWriter;
import java.net.MalformedURLException;
import java.net.URL;

import javax.xml.bind.JAXBContext;
import javax.xml.bind.JAXBException;
import javax.xml.bind.Marshaller;

import org.junit.Before;
import org.junit.Test;

import genius.core.repository.ProfileRepItem;

public class ProfileRepItemTest {

	private ProfileRepItem normalItem;
	private String normalString = "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?><profileRepItem url=\"file:test\"/>";

	@Before
	public void before() throws MalformedURLException {
		normalItem = new ProfileRepItem(new URL("file:test"), null);
	}

	@Test
	public void testNormalItem() throws JAXBException, MalformedURLException {

		StringWriter writer = new StringWriter();

		JAXBContext context = JAXBContext.newInstance(ProfileRepItem.class);
		Marshaller m = context.createMarshaller();
		m.marshal(normalItem, writer);

		assertEquals(normalString, writer.toString());
		System.out.println(writer.toString());

	}
}
